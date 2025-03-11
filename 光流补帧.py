import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_flow(prev_frame, next_frame):
    """
    使用 Farneback 算法计算稠密光流
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Farneback 参数（可根据视频调整）
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5,  # 金字塔缩放因子
        levels=3,  # 金字塔层数
        winsize=15,  # 窗口大小
        iterations=3,  # 迭代次数
        poly_n=5,  # 像素邻域大小
        poly_sigma=1.2,  # 高斯标准差
        flags=0  # 光流模式
    )
    return flow  # 形状 (H, W, 2) 的 numpy 数组


def warp_image_opencv(img, flow):
    """
    OpenCV 实现的图像变形（参考用）
    """
    h, w = img.shape[:2]
    flow_map = -flow.copy()
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)


def warp_image_pytorch(img, flow, t=0.5):
    """
    PyTorch GPU 加速的图像变形
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将图像转换为 PyTorch 张量
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]

    # 生成坐标网格
    h, w = img.shape[:2]
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid = torch.stack([grid_x, grid_y], dim=-1).float().to(device)  # [H, W, 2]

    # 调整光流（考虑时间参数 t）
    adjusted_flow = torch.from_numpy(flow).to(device) * t
    warped_grid = grid - adjusted_flow

    # 归一化到 [-1, 1]
    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w - 1) - 1.0  # x 坐标
    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h - 1) - 1.0  # y 坐标

    # 执行双线性采样
    warped = F.grid_sample(
        img_tensor,
        warped_grid.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # 转换回 numpy 数组
    result = warped.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return result


def interpolate_frames(prev_frame, next_frame, t=0.5):
    """
    生成中间帧（核心算法）
    """
    # 计算双向光流
    flow_forward = compute_flow(prev_frame, next_frame)
    flow_backward = compute_flow(next_frame, prev_frame)

    # 变形图像
    warped_prev = warp_image_pytorch(prev_frame, flow_forward, t)
    warped_next = warp_image_pytorch(next_frame, flow_backward, (1 - t))

    # 线性混合
    intermediate = cv2.addWeighted(warped_prev, (1 - t), warped_next, t, 0)
    return intermediate


def video_interpolation(input_path, output_path, interval=1):
    """
    视频插帧主函数
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * (interval + 1), (width, height))

    ret, prev = cap.read()
    while ret:
        out.write(prev)  # 写入原始帧

        # 读取下一帧
        ret, next = cap.read()
        if not ret:
            break

        # 生成中间帧
        intermediate = interpolate_frames(prev, next)
        out.write(intermediate)

        prev = next

    cap.release()
    out.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--interval', type=int, default=1, help='每帧插入的中间帧数量')
    args = parser.parse_args()

    video_interpolation(args.input, args.output, args.interval)