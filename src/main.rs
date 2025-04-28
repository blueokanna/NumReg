#[cfg(target_os = "windows")]
extern crate winapi;
use flate2::read::GzDecoder;
use std::ffi::CString;
use std::io::Write;
use std::{fs, io::{self, copy}, path::{Path, PathBuf}};
use tch::vision::mnist;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

const TRAIN_IMGS_GZ: &[u8]  = include_bytes!("../data/train-images-idx3-ubyte.gz");
const TRAIN_LABS_GZ: &[u8]  = include_bytes!("../data/train-labels-idx1-ubyte.gz");
const TEST_IMGS_GZ:  &[u8]  = include_bytes!("../data/t10k-images-idx3-ubyte.gz");
const TEST_LABS_GZ:  &[u8]  = include_bytes!("../data/t10k-labels-idx1-ubyte.gz");

fn write_embedded_gz(fname: &str, data: &[u8]) -> io::Result<()> {
    let path = Path::new("data").join(fname);
    if path.exists() {
        return Ok(());
    }
    println!("写入嵌入文件 → {:?}", path);
    fs::create_dir_all("data")?;
    let mut f = fs::File::create(&path)?;
    f.write_all(data)?;
    Ok(())
}

/// 如果存在 `<src>.gz` 且 `<src>` 不存在，就解压它。
fn decompress_if_gz(src_gz: &Path) -> io::Result<()> {
    let dst = src_gz.with_extension(""); // 去掉 .gz
    if dst.exists() {
        return Ok(());
    }
    println!("解压 {:?} → {:?}", src_gz, dst);
    let mut gz = GzDecoder::new(fs::File::open(src_gz)?);
    let mut out = fs::File::create(&dst)?;
    copy(&mut gz, &mut out)?;
    Ok(())
}

#[cfg(target_os = "windows")]
use winapi::um::libloaderapi::LoadLibraryA;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1) 设备
    let path = CString::new("D:\\libtorch\\lib\\torch_cuda.dll").unwrap();
    unsafe {
        LoadLibraryA(path.into_raw());
    }
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CuDNN available: {}", tch::Cuda::cudnn_is_available());
    let device = if Device::cuda_if_available().is_cuda() {
        println!("Using CUDA");
        Device::Cuda(0)
        //Device::Cpu
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };

    write_embedded_gz("train-images-idx3-ubyte.gz", TRAIN_IMGS_GZ)?;
    write_embedded_gz("train-labels-idx1-ubyte.gz", TRAIN_LABS_GZ)?;
    write_embedded_gz("t10k-images-idx3-ubyte.gz",  TEST_IMGS_GZ)?;
    write_embedded_gz("t10k-labels-idx1-ubyte.gz",  TEST_LABS_GZ)?;

    for fname in &[
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ] {
        decompress_if_gz(&PathBuf::from("data").join(fname))?;
    }

    println!("\n加载 MNIST 数据集 …");
    let m = mnist::load_dir("data")?;
    println!("MNIST 加载完毕\n");

    // ============ MLP ============
    let vs_mlp = nn::VarStore::new(device);
    let root_mlp = &vs_mlp.root();
    let fc1 = nn::linear(root_mlp / "fc1", 784, 128, Default::default());
    let fc2 = nn::linear(root_mlp / "fc2", 128, 64, Default::default());
    let fc3 = nn::linear(root_mlp / "fc3", 64, 10, Default::default());

    // 打印 MLP 结构与参数量
    let dummy_mlp = Tensor::zeros(&[1, 784], (Kind::Float, device));
    println!("--- MLP 网络层结构与参数量 ---");
    println!("Input: {:?}", dummy_mlp.size());
    let h1 = dummy_mlp.apply(&fc1).relu();
    println!(
        "After fc1: {:?} | params = {}",
        h1.size(),
        fc1.ws.numel() as i64 + fc1.bs.as_ref().unwrap().numel() as i64
    );
    let h2 = h1.apply(&fc2).relu();
    println!(
        "After fc2: {:?} | params = {}",
        h2.size(),
        fc2.ws.numel() as i64 + fc2.bs.as_ref().unwrap().numel() as i64
    );
    let o3 = h2.apply(&fc3);
    println!(
        "After fc3: {:?} | params = {}",
        o3.size(),
        fc3.ws.numel() as i64 + fc3.bs.as_ref().unwrap().numel() as i64
    );
    let total_mlp_params: i64 = vs_mlp
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum();
    println!("MLP 总参数量: {}\n", total_mlp_params);

    let forward_mlp = |xs: &Tensor| xs.apply(&fc1).relu().apply(&fc2).relu().apply(&fc3);
    let mut opt_mlp = nn::AdamW::default().build(&vs_mlp, 1e-3)?;

    // 训练 MLP
    for epoch in 1..=10 {
        let (mut loss_sum, mut correct, mut total) = (0.0, 0, 0);
        for (bx, by) in m.train_iter(64).shuffle() {
            let x = bx.view([-1, 784]).to_device(device);
            let y = by.to_device(device);
            let logits = forward_mlp(&x);
            let loss = logits.cross_entropy_for_logits(&y);
            opt_mlp.backward_step(&loss);

            loss_sum += loss.double_value(&[]) * y.size()[0] as f64;
            let bc = logits
                .argmax(-1, false)
                .eq_tensor(&y)
                .sum(Kind::Int64)
                .int64_value(&[]);
            correct += bc;
            total += y.size()[0];
        }
        println!(
            "Epoch {:2} | MLP loss {:>7.4} | acc {:>5.2}%",
            epoch,
            loss_sum / total as f64,
            100.0 * correct as f64 / total as f64
        );
    }

    // 测试 MLP
    let (mut correct, mut total) = (0, 0);
    for (bx, by) in m.test_iter(1000) {
        let x = bx.view([-1, 784]).to_device(device);
        let y = by.to_device(device);
        let logits = forward_mlp(&x);
        let bc = logits
            .argmax(-1, false)
            .eq_tensor(&y)
            .sum(Kind::Int64)
            .int64_value(&[]);
        correct += bc;
        total += y.size()[0];
    }
    let mlp_acc = 100.0 * correct as f64 / total as f64;
    println!("MLP 测试准确率: {:>5.2}%\n", mlp_acc);

    // ============ CNN ============
    let vs_cnn = nn::VarStore::new(device);
    let root_cnn = &vs_cnn.root();
    let conv1 = nn::conv2d(root_cnn / "conv1", 1, 32, 5, Default::default());
    let conv2 = nn::conv2d(root_cnn / "conv2", 32, 64, 5, Default::default());
    let fc1_c = nn::linear(root_cnn / "fc1", 64 * 4 * 4, 512, Default::default());
    let fc2_c = nn::linear(root_cnn / "fc2", 512, 10, Default::default());

    // 打印 CNN 结构与参数量
    let dummy_cnn = Tensor::zeros(&[1, 1, 28, 28], (Kind::Float, device));
    println!("--- CNN 网络层结构与参数量 ---");
    println!("Input: {:?}", dummy_cnn.size());
    let c1 = dummy_cnn.apply(&conv1).relu();
    println!(
        "After conv1: {:?} | params = {}",
        c1.size(),
        conv1.ws.numel() as i64 + conv1.bs.as_ref().unwrap().numel() as i64
    );
    let p1 = c1.max_pool2d_default(2);
    println!("After pool1: {:?}", p1.size());
    let c2 = p1.apply(&conv2).relu();
    println!(
        "After conv2: {:?} | params = {}",
        c2.size(),
        conv2.ws.numel() as i64 + conv2.bs.as_ref().unwrap().numel() as i64
    );
    let p2 = c2.max_pool2d_default(2);
    println!("After pool2: {:?}", p2.size());
    let flat = p2.view([-1, 64 * 4 * 4]);
    println!("After flatten: {:?}", flat.size());
    let h1_c = flat.apply(&fc1_c).relu();
    println!(
        "After fc1: {:?} | params = {}",
        h1_c.size(),
        fc1_c.ws.numel() as i64 + fc1_c.bs.as_ref().unwrap().numel() as i64
    );
    let out_c = h1_c.apply(&fc2_c);
    println!(
        "After fc2: {:?} | params = {}",
        out_c.size(),
        fc2_c.ws.numel() as i64 + fc2_c.bs.as_ref().unwrap().numel() as i64
    );
    let total_cnn_params: i64 = vs_cnn
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum();
    println!("CNN 总参数量: {}\n", total_cnn_params);

    let forward_cnn = |xs: &Tensor| {
        xs.apply(&conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 64 * 4 * 4])
            .apply(&fc1_c)
            .relu()
            .apply(&fc2_c)
    };
    let mut opt_cnn = nn::Adam::default().build(&vs_cnn, 1e-3)?;

    // 训练 CNN
    for epoch in 1..=5 {
        let (mut loss_sum, mut correct, mut total) = (0.0, 0, 0);
        for (bx, by) in m.train_iter(64).shuffle() {
            // **关键修改**：直接 reshape 为 [N,1,28,28]
            let x = bx.view([-1, 1, 28, 28]).to_device(device);
            let y = by.to_device(device);
            let logits = forward_cnn(&x);
            let loss = logits.cross_entropy_for_logits(&y);
            opt_cnn.backward_step(&loss);

            loss_sum += loss.double_value(&[]) * y.size()[0] as f64;
            let bc = logits
                .argmax(-1, false)
                .eq_tensor(&y)
                .sum(Kind::Int64)
                .int64_value(&[]);
            correct += bc;
            total += y.size()[0];
        }
        println!(
            "Epoch {:2} | CNN loss {:>7.4} | acc {:>5.2}%",
            epoch,
            loss_sum / total as f64,
            100.0 * correct as f64 / total as f64
        );
    }

    // 测试 CNN
    let (mut correct, mut total) = (0, 0);
    for (bx, by) in m.test_iter(1000) {
        let x = bx.view([-1, 1, 28, 28]).to_device(device);
        let y = by.to_device(device);
        let logits = forward_cnn(&x);
        let bc = logits
            .argmax(-1, false)
            .eq_tensor(&y)
            .sum(Kind::Int64)
            .int64_value(&[]);
        correct += bc;
        total += y.size()[0];
    }
    let cnn_acc = 100.0 * correct as f64 / total as f64;
    println!("CNN 测试准确率: {:>5.2}%\n", cnn_acc);

    // 最终对比
    println!(
        "====== 对比 ======\nMLP 准确率: {:>5.2}%\nCNN 准确率: {:>5.2}%",
        mlp_acc, cnn_acc
    );

    Ok(())
}
