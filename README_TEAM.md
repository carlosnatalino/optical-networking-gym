# Hướng dẫn chạy chương trình mô phỏng của nhóm

Khung chương trình thuộc BTL MVT, xây dựng trên [Optical Networking Gym](https://github.com/carlosnatalino/optical-networking-gym). Chương trình tạo môi trường mạng quang, chọn action bằng thuật toán First-Fit, chạy một episode và lưu kết quả JSON.

## 1. Chuẩn bị và cài đặt

Hướng dẫn sử dụng Windows, PowerShell và môi trường ảo `.venv`. Đặt file README này tại thư mục gốc repository, cùng cấp với `pyproject.toml`.

Nếu đã cài đặt và chạy thành công ví dụ gốc, chuyển đến mục 3.

Trên máy mới, cài Git, Python và bộ biên dịch C/C++ phù hợp để build phần mở rộng Cython (trên Windows có thể dùng Visual Studio Build Tools với thành phần Desktop development with C++). Python 3.12 là phiên bản theo hướng dẫn cài đặt của nhóm; cần đối chiếu yêu cầu của phiên bản repository đang sử dụng.

Clone repository của nhóm nếu đã có; nếu chưa, có thể bắt đầu từ repository gốc:

```powershell
git clone https://github.com/carlosnatalino/optical-networking-gym.git
cd optical-networking-gym
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Khi clone repository gốc, cần bổ sung thư mục `team/` của nhóm trước khi chạy chương trình nhóm.

Kiểm tra cài đặt:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe examples/quickstart/basic_first_fit.py
```

Trong VS Code, dùng **Python: Select Interpreter** và chọn `.venv\Scripts\python.exe`. Các lệnh trong tài liệu gọi Python trực tiếp nên không cần kích hoạt môi trường ảo.

## 2. Cấu trúc chương trình

| Đường dẫn | Vai trò |
|---|---|
| `team/__init__.py` | File rỗng, đánh dấu package `team` |
| `team/main.py` | Cấu hình lần chạy, tạo agent và môi trường, gọi runner, lưu JSON |
| `team/env_factory.py` | Hàm `create_env(config)` khởi tạo môi trường |
| `team/runner.py` | Hàm `run_episode(env, agent, seed)` thực hiện vòng lặp và trả về chỉ số |
| `team/agents/__init__.py` | File rỗng, đánh dấu package thuật toán |
| `team/agents/first_fit.py` | Lớp `FirstFitAgent`, thuật toán mẫu |
| `team/results/` | Chứa các file JSON; tự tạo khi lưu kết quả |
| `team_main.py` | Bản chương trình ban đầu để đối chiếu |

Luồng chạy: `main.py` tạo môi trường và agent → `run_episode()` gọi `reset()` → agent chọn action → môi trường thực hiện `step()` → lặp đến khi episode kết thúc → lưu kết quả.

## 3. Chạy chương trình nhóm

Mở PowerShell tại thư mục gốc repository. Ví dụ trên máy đã kiểm tra:

```powershell
cd D:\BTL-MVT\optical-networking-gym
.\.venv\Scripts\python.exe -m team.main
```

Dùng `-m team.main` để chạy module trong package. Chương trình in tiến độ mỗi 10 bước và in kết quả cuối episode.

## 4. Cấu hình mô phỏng

Sửa dictionary `config` trong `team/main.py`:

```python
config = {
    "seed": 42,
    "load": 300.0,
    "episode_length": 100,
}
```

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `seed` | `42` | Hạt giống ngẫu nhiên phục vụ tái lập |
| `load` | `300.0` | Tải lưu lượng, đơn vị Erlang |
| `episode_length` | `100` | Số bước cấu hình cho một episode |

Các tham số còn lại được đặt trong `team/env_factory.py`:

| Tham số | Giá trị |
|---|---|
| `scenario` | `nobel_eu_baseline` |
| `topology_id` trong `overrides` | `nobel-eu` |
| `num_spectrum_resources` | `320` |
| `k_paths` | `3` |
| `modulations_to_consider` | `4` |
| `bit_rates` | `(10, 40, 100, 400)` |
| `modulations` | `BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM` |

Hiện tại JSON chỉ lưu ba tham số trong `config`, chưa lưu toàn bộ tham số cố định của factory. Khi thay đổi cấu hình mạng để thực nghiệm, cần ghi lại các thay đổi hoặc mở rộng phần cấu hình được xuất.

## 5. Giao diện thuật toán

Mỗi agent cung cấp tên và phương thức:

```python
class CustomAgent:
    name = "custom_agent"

    def solution(self, obs, info, env):
        # Chọn và trả về action theo giao diện của môi trường.
        raise NotImplementedError
```

| Đầu vào | Nội dung |
|---|---|
| `obs` | Quan sát hiện tại |
| `info` | Thông tin bổ sung từ `reset()` hoặc `step()` |
| `env` | Môi trường, dùng khi cần lấy action mask hoặc ngữ cảnh thuật toán |

First-Fit hiện lấy mask từ `info.get("mask")`; nếu không có thì gọi `env.action_masks()`, sau đó dùng `select_first_fit_action(mask)`.

Để tích hợp thuật toán mới, tạo file trong `team/agents/`, import lớp vào `team/main.py` và thay dòng khởi tạo `agent`. Giữ nguyên vòng lặp trong `runner.py` nếu giao diện agent không đổi.

Runner kiểm tra `env.action_space.contains(action)` trước khi gọi `step()`. Kiểm tra này xác nhận action thuộc không gian hành động; việc cấp phát thành công vẫn phụ thuộc xử lý của môi trường.

## 6. Vòng lặp môi trường

```python
obs, info = env.reset(seed=seed)
action = agent.solution(obs, info, env)
obs, reward, terminated, truncated, info = env.step(action)
```

`reset()` khởi tạo episode. `step()` thực hiện quyết định và trả về quan sát mới, reward, hai cờ kết thúc và thông tin bổ sung. Runner dừng khi `terminated or truncated` đúng. `main.py` gọi `env.close()` trong `finally` để đóng môi trường cả khi xảy ra lỗi.

## 7. Kết quả JSON

Kết quả nằm trong `team/results/`, với tên dạng `first_fit_<thời gian UTC>.json`. Hậu tố `Z` biểu thị UTC; giờ Việt Nam bằng UTC cộng 7 giờ. Mỗi lần chạy tạo một file mới.

Ví dụ thực tế:

```json
{
  "algorithm": "first_fit",
  "config": {
    "seed": 42,
    "load": 300.0,
    "episode_length": 100
  },
  "metrics": {
    "steps": 100,
    "total_reward": 96.27289358129727,
    "request_blocking_ratio": 0.01,
    "bit_rate_blocking_ratio": 0.03289473684210526
  }
}
```

| Trường | Ý nghĩa |
|---|---|
| `algorithm` | Tên agent |
| `config` | Cấu hình lần chạy được xuất |
| `steps` | Số bước thực hiện |
| `total_reward` | Tổng reward trong episode |
| `request_blocking_ratio` | Tỷ lệ yêu cầu bị chặn, lấy từ `episode_service_blocking_rate` |
| `bit_rate_blocking_ratio` | Tỷ lệ bit rate bị chặn, lấy từ `episode_bit_rate_blocking_rate` |

Các tỷ lệ được lưu dưới dạng số từ 0 đến 1; nhân 100 để đổi sang phần trăm. Ví dụ trên có tỷ lệ chặn yêu cầu 1% và tỷ lệ chặn bit rate khoảng 3,2895%. Hai chỉ số có thể khác nhau vì các yêu cầu có bit rate khác nhau.

Reward phụ thuộc cách môi trường định nghĩa hàm thưởng; không diễn giải tổng reward trực tiếp thành tỷ lệ cấp phát thành công.


## 8. Lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| `No module named team` | Chạy từ thư mục gốc repository; kiểm tra `team/main.py` và các file `__init__.py` |
| `No module named optical_networking_gym` | Dùng đúng `.venv` và chạy lại lệnh cài editable trong mục 1 |
| Lỗi liên quan module Cython | Kiểm tra compiler và build lại bằng `setup.py build_ext --inplace` trong đúng môi trường |
| Không thấy JSON | Kiểm tra `team/results/` và đường dẫn chương trình in ra; nếu có traceback, xử lý lỗi trước |
| Tỷ lệ chặn lớn hơn 0 | Đây có thể là kết quả mô phỏng bình thường; kiểm tra cấu hình và thuật toán để phân tích |

## 9. Ghi nhận phiên bản và quản lý mã nguồn

Ghi nhận commit và các gói đang cài để hỗ trợ tái lập:

```powershell
git rev-parse HEAD
.\.venv\Scripts\python.exe -m pip freeze > requirements-team.txt
```

`requirements-team.txt` là bản chụp môi trường hiện tại; cần kiểm tra các dòng đường dẫn cục bộ hoặc editable trước khi dùng trên máy khác.

Mỗi thành viên làm trên nhánh riêng. Không đưa `.venv`, cache và file build vào Git. Thống nhất cách lưu kết quả thực nghiệm với nhóm; giữ nguyên giấy phép và ghi nhận nguồn Optical Networking Gym.
