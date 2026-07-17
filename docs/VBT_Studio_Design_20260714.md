# VBT Studio 动态渲染器设计

日期：2026-07-14  
状态：设计冻结，等待进入实现阶段

## 1. 目标

VBT Studio 是面向动态 VBT 体数据的桌面渲染与论文数据生产工具。它不是单帧
图片查看器，也不是 Blender 脚本的简单外壳。

核心目标：

1. 点击或拖入 `.vbtp` 后直接加载，不生成 OpenVDB/NanoVDB 代理。
2. VBT 保持 GPU 常驻，时间轴变化只更新帧参数，不重复上传完整文件。
3. 支持播放、循环、逐帧、指定帧和范围导出。
4. 支持 density、flames、temperature 等多字段组合。
5. 支持烟雾、火焰、科学场和后续流体的不同渲染模型。
6. 支持材质、传递函数、光照、相机和采样参数实时调整。
7. 支持 PNG、EXR、图片序列和 MP4 导出。
8. 支持原始/重建 A/B、滑动分割、差异图和论文批处理。
9. 所有导出带可复现 JSON sidecar，记录文件哈希、帧、相机、材质和 GPU。

## 2. 当前事实与约束

### 2.1 数据资产

本地完整源序列只有七组。含真实火焰字段的只有：

- Ground Explosion：`density + flames`
- Aerial Explosion：`density + flames + temperature`

其余五组只有 density。旧论文目录中的 Small Campfire 等图片没有对应的完整源
VDB 序列，不能作为可重跑的压缩数据集。

新增火焰数据集必须满足：

- 至少 64 帧，稳定 index-to-world 变换；
- 必须有 density 和 flames，temperature 可选；
- 帧编号连续，空帧和损坏帧有清单；
- 数据许可允许论文使用；
- 先生成 metadata 和完整性报告，再进入压缩。

建议补充四类数据：campfire、torch/jet flame、fireball、surface fire。不同类别
可以覆盖稳定燃烧、定向喷射、快速爆燃和贴地传播。

### 2.2 现有 GPU 后端

现有 `render/` 是 Vulkan compute benchmark，不是窗口渲染器。它已经具备最重要
的基础：VBT offset/payload 常驻 GPU，shader 在任意 `(x,y,z,t)` 直接采样。

Industrial Chimney 的已验证基准：

- 640 x 360，192 steps；
- 230,400 rays；
- GPU dispatch 平均 8.514 ms；
- 约 5.196 Gsamples/s；
- 初次上传 38.371 ms；
- 切换帧上传时间 0 ms；
- Vulkan/CPU mismatch 为 0。

当前 shader 只输出每条 ray 的密度积分，没有颜色、光照、前向合成、交换链或
交互相机。它应作为 VBT Studio 的采样核心，而不是直接包装成最终产品。

### 2.3 现有 native Cycles 后端

patched Blender/Cycles 已能直接加载 `.vbtp` 并在 CPU/CUDA 中采样，适合高质量
最终导出。但是当前 patch 把 `frame_index` 写进 resident blob，并把 frame 放进
`VBTImageLoader::equals()`。动画换帧可能产生新的 image identity 并重新上传完整
blob，不适合实时播放。

进入动画导出前必须改为：

- offsets 和 payload 是 immutable resident resource；
- frame index 是独立的小型 device parameter；
- 换帧只更新参数，不创建新 loader，不重传 VBT；
- density/flames/temperature 使用多属性绑定，而不是只有 density。

## 3. 总体架构

采用双后端，而不是在“实时”与“质量”之间二选一。

```text
                        VBT Studio
  +-------------------------------------------------------+
  | Asset/Project Layer                                   |
  | .vbtp + metadata + .vbtset + .vbtproj                 |
  +--------------------------+----------------------------+
                             |
             +---------------+----------------+
             |                                |
  +----------v-----------+         +----------v-----------+
  | Vulkan Interactive   |         | Production Export    |
  | resident VBT buffers |         | Vulkan accumulation  |
  | dynamic timeline     |         | optional native      |
  | material preview     |         | Cycles worker        |
  +----------+-----------+         +----------+-----------+
             |                                |
  +----------v--------------------------------v-----------+
  | PNG / EXR / sequence / MP4 / JSON / CSV / contact     |
  +-------------------------------------------------------+
```

### 技术选择

- C++20
- Vulkan 1.3
- GLFW：窗口、输入和 Vulkan surface
- Dear ImGui docking：专业工具界面
- Vulkan Memory Allocator：显存分配
- nlohmann/json：项目、材质和导出配置
- stb_image_write：PNG
- TinyEXR：线性 HDR 导出
- FFmpeg 子进程：MP4 编码
- CMake：与当前仓库构建系统一致

不选择 Web 前端：超大 VBT、Vulkan SSBO、CUDA/Cycles 和本地文件访问不适合浏览
器。也不只做 Blender addon：Blender 适合最终渲染，但当前 frame reload 行为和
viewport 路径不满足高性能动态播放。

## 4. 数据模型

### 4.1 `.vbtset`

单个 `.vbtp` 是一个标量场。烟火需要把多个 VBT 组合成一个可渲染资产。

```json
{
  "version": 1,
  "name": "Aerial Explosion",
  "frames": 120,
  "fps": 24,
  "fields": {
    "density": {
      "vbt": "aerial_explosion_density.vbtp",
      "metadata": "aerial_explosion_density.metadata.json"
    },
    "flames": {
      "vbt": "aerial_explosion_flames.vbtp",
      "metadata": "aerial_explosion_flames.metadata.json"
    }
  },
  "defaults": {
    "material": "fire_warm",
    "frame": 60
  }
}
```

直接打开单个 `.vbtp` 时，程序自动寻找同名 metadata，并创建临时单字段 asset。

### 4.2 `.vbtproj`

项目文件保存：

- 相对 asset 路径；
- 相机位置、目标、焦距和裁剪；
- 时间轴、FPS、循环模式；
- 材质实例参数；
- 背景、灯光和质量设置；
- A/B reference 配置；
- 导出预设。

项目文件不复制大体积 VBT。

### 4.3 Material JSON

材质不是写死在 shader 中。每个 preset 是 JSON 参数集合，可复制并保存为用户
preset。

首批内置材质：

- `paper_gray`
- `charcoal`
- `soft_ash`
- `cool_steel`
- `fire_warm`
- `fire_embers`
- `fire_blackbody`（需要 temperature）

## 5. Vulkan 实时渲染核心

### 5.1 Resident Asset

每个字段维护：

- immutable header；
- immutable leaf offset table；
- immutable payload pages；
- metadata transform；
- role：density/flames/temperature/scientific/level-set。

帧切换只更新 push constant 或 uniform：

```text
frame, fractional_time, quality, camera, material parameters
```

大文件不能永远依赖单 SSBO。Dust Shockwave 已接近 4 GB。资源层从第一版就使用
page abstraction：offset 解码为 page id + byte offset，descriptor indexing 绑定
多个 256 MB payload page。显存不足时再加入 LRU page streaming，而不是重写
shader 接口。

### 5.2 体渲染积分

compute shader 输出 RGBA16F storage image，使用 front-to-back emission-
absorption：

```text
sigma_t = density_scale * max(density, 0)
alpha   = 1 - exp(-sigma_t * dt)
color  += transmittance * (scatter + emission) * alpha
transmittance *= 1 - alpha
```

火焰 emission 来自 flames transfer function；有 temperature 时可选 blackbody。
烟雾和火焰在同一 ray march 中采样，避免两次完整穿越。

### 5.3 性能策略

按优先级实现：

1. VBT payload 常驻，帧切换零上传。
2. Empty leaf DDA 跳跃，不对空块逐步采样。
3. transmittance 小于阈值时 early termination。
4. 动态分辨率 50%/67%/100%。
5. 静止相机 temporal accumulation，交互时自动降采样。
6. blue-noise ray jitter，减少低 step 数条带。
7. descriptor indexing 支持多字段和 payload page。
8. 可用时采用 dedicated compute queue + timeline semaphore。

质量档位：

- Interactive：动态分辨率，64-128 steps，无阴影或短 shadow ray。
- Balanced：128-256 steps，单主光、短 shadow march。
- Final Vulkan：256-1024 steps，多帧 accumulation，原生分辨率。
- Final Cycles：离线路径追踪，用于最终论文 hero image。

性能目标基于现有 8.514 ms/640x360/192-step 基准，而不是空泛承诺：

- 720p Interactive：目标 45-60 FPS；
- 1080p Balanced：目标 24-45 FPS，依数据稀疏度变化；
- frame advance CPU overhead：小于 0.2 ms，不上传 VBT；
- UI frame 不等待磁盘或编码任务。

### 5.4 渲染 pass

```text
Ray Setup -> VBT Volume Integrate -> Optional Shadow -> Tone Map -> UI Composite
```

论文差异模式增加：

```text
Reference Render + VBT Render -> Split/Wipe/Difference/Heatmap
```

## 6. 界面设计

界面是 DCC/科学可视化工具，不做营销首页。

### 顶部工具栏（固定 40 px）

- Open、Save Project、Undo、Redo
- Play/Pause、Previous Frame、Next Frame
- Camera reset、Fit volume
- Screenshot、Export
- Preview quality segmented control

使用 Lucide 风格图标并提供 tooltip；不使用大号文字按钮。

### 左侧 Asset/Scene（280 px，可折叠）

- Asset tree：density、flames、temperature 字段及状态
- Reference source：可选 VDB sequence / second VBT
- Camera、Light、Background scene nodes
- 每个字段显示尺寸、帧数、VBT 大小、显存占用和可见开关

### 中央 Viewport

- 占据最大面积，不放进装饰卡片
- Orbit/Pan/Dolly
- 单视图、左右 A/B、wipe、difference 四种模式
- 左上角仅显示帧、FPS、GPU ms、step 数
- 右上角显示 progressive accumulation sample

### 右侧 Inspector（340 px，可折叠）

Tabs：Material、Render、Camera、Export。

Material：

- preset 下拉菜单；
- density scale、threshold、gamma；
- color swatch；
- anisotropy；
- flame strength、flame threshold；
- 可编辑 color ramp；
- temperature/blackbody toggle；
- background 和曝光。

Render：

- quality segmented control；
- step count/step size；
- dynamic resolution toggle；
- shadow toggle 和 shadow steps；
- temporal accumulation；
- renderer：Vulkan / Cycles export。

### 底部时间轴（固定 92 px）

- frame scrubber；
- current/total frame 数字输入；
- FPS；
- loop / ping-pong；
- integer-frame / interpolated playback；
- range in/out；
- key frame markers 和空 flames frame 标记。

控件尺寸固定，播放时标签和帧数变化不能推动布局。

### 状态栏（24 px）

- GPU 名称；
- resident VRAM / budget；
- asset load state；
- active backend；
- background export progress。

## 7. 动画与时间

VBT 原始数据是离散帧。默认使用 integer frame，保证论文可复现。

播放模式：

- Real-time：按 asset FPS；落后时跳显示帧，但不改变导出。
- Every frame：不跳帧，适合检查。
- Interpolated：shader 使用 float time 在相邻时间样本间插值，仅用于视觉播放。
- Selected frame：固定帧，高质量 accumulation。

时间轴变化不重新创建 asset、不重新编译 shader、不重新上传 payload。

## 8. 导出系统

### 单帧

- PNG 8/16-bit
- EXR half/float
- 可选透明背景
- 可选 UI overlay

### 动画

- PNG/EXR sequence
- MP4 H.264/H.265
- 指定 FPS、范围、分辨率和后端
- 导出在后台线程/worker 运行

### 可复现 sidecar

每次导出写同名 JSON：

- VBT 和 metadata SHA256；
- renderer git revision；
- GPU/driver；
- 帧或范围；
- 相机矩阵；
- 所有材质参数；
- step、samples、分辨率；
- load/render/export timing。

## 9. 论文数据生产模式

VBT Studio 内置 `Paper Campaign` job，不靠手工反复点击。

### 图像数据

每个数据集生成：

1. `t-1 / t / t+1` 三帧条带；
2. 每帧 Original / Reconstructed；
3. 相同相机、材质、分辨率和采样；
4. 烟雾四种材质；
5. 火焰三种材质：warm、embers、blackbody（有 temperature 时）；
6. difference heatmap；
7. 数据集总览 contact sheet；
8. 动态 MP4 对比。

展示图和数值图分开：展示图允许艺术材质；PSNR/SSIM 图必须使用固定线性传递
函数和完全一致的参数。

### 表数据

输出 JSON + CSV + LaTeX：

- source VDB bytes；
- dense raw bytes；
- VBT bytes；
- raw/VBT 和 VDB/VBT 压缩比；
- voxel PSNR、RMSE、MAE、MaxAbs；
- render PSNR、SSIM、MAE；
- 每帧 mean/std/min/max 和全序列 mean/std/min/max；
- mode 0/1/2/3 leaf 数和占比；
- load ms、resident upload ms、frame-switch ms；
- GPU render ms、FPS、samples/s；
- VRAM resident bytes。

不能只报告一个代表帧。代表帧用于图，表格至少提供全序列统计和最差帧。

## 10. Native Cycles 生产后端

Vulkan 是交互主后端。Cycles 作为 worker，不阻塞主 UI：

1. VBT Studio 将 project snapshot 写入临时 job JSON；
2. patched Blender worker 读取同一 camera/material/timeline；
3. worker 使用 native VBT，不生成 VDB proxy；
4. 进度通过 stdout JSON lines 返回；
5. 完成后写 image/sequence 和 sidecar。

必须先完成 Cycles patch 的 persistent payload 和多字段 attribute 改造。否则动画
每帧可能重复加载完整 VBT，违背性能目标。

## 11. 仓库结构

建议新增：

```text
viewer/
  CMakeLists.txt
  src/app/
  src/assets/
  src/render/
  src/ui/
  shaders/
  presets/materials/
  tests/
docs/VBT_Studio_Design_20260714.md
scripts/run_paper_campaign.py
```

采样解码必须从 `src/render_temporal_*` 和现有 Vulkan shader 抽出共享实现，避免
viewer、benchmark 和 Cycles 三份代码继续漂移。

## 12. 实施阶段与验收

### Phase 1：可用动态 Viewer

- 打开单字段 `.vbtp`；
- GPU resident load；
- orbit camera；
- 时间轴播放和指定帧；
- smoke 材质；
- PNG 导出；
- FPS/GPU timing。

验收：Industrial Chimney 连续播放，帧切换零 payload upload，导出帧与离线 Vulkan
采样一致。

### Phase 2：多字段烟火与性能

- `.vbtset`；
- density + flames + temperature；
- fire transfer function；
- empty-leaf skip、early exit、dynamic resolution；
- payload pages。

验收：Ground/Aerial 动画可播放，火焰和烟雾空间对齐；720p Interactive 达到目标
帧率区间；大于 4 GB VBT 不依赖单 buffer 假设。

### Phase 3：对比与论文数据

- VDB/NanoVDB reference cache；
- A/B、wipe、heatmap；
- batch image/metric/table；
- PNG/EXR/MP4；
- JSON/CSV/LaTeX。

验收：一条 campaign 配置可重建当前七组烟雾、两组火焰的图表，不需要手工改路径。

### Phase 4：Native Cycles Worker

- persistent payload；
- frame parameter update；
- multi-field attributes；
- background queue；
- high-sample animation export。

验收：连续帧导出不重复上传完整 VBT；Vulkan 与 Cycles 使用同一 project/material
参数；导出 sidecar 完整。

### Phase 5：流体和发布

- level-set iso surface path；
- water material；
- installer、crash report、sample project；
- UI 截图和论文系统图。

## 13. 首个实现切片

第一个代码切片不做假 UI，应直接完成以下闭环：

1. GLFW + Vulkan + ImGui 窗口；
2. 打开 `.vbtp`；
3. offset/payload GPU resident；
4. 现有 VBT shader 接入 RGBA16F ray march；
5. 相机 orbit；
6. frame slider 和播放；
7. density scale/color/background；
8. PNG + JSON sidecar；
9. GPU timestamp 和 frame-switch upload 统计。

这个切片完成后，再加 flames 多字段。这样底层资源、时间轴和导出接口从第一天就
是正确的，不会先做一个只能看单帧的界面再推倒重来。
