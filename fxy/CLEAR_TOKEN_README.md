## Antigravity Token 清空工具

用于清空 Antigravity (`state.vscdb`) 中的登录 Token，实现账号登出。

### 📋 文件说明

#### 1. **Python 脚本** (推荐)

- **文件**: `clear_antigravity_token.py`
- **适用**: macOS, Windows, Linux
- **依赖**: Python 3.6+（内置 sqlite3 和 base64）
- **用法**:
    ```bash
    python3 clear_antigravity_token.py
    ```

**特点**:

- ✅ 跨平台支持
- ✅ 自动备份原文件
- ✅ 彩色提示信息
- ✅ 操作确认机制
- ✅ 详细的 Protobuf 解析

#### 2. **Bash 脚本**

- **文件**: `clear_antigravity_token.sh`
- **适用**: macOS, Linux
- **用法**:
    ```bash
    chmod +x clear_antigravity_token.sh
    ./clear_antigravity_token.sh
    ```

**特点**:

- ✅ 快速执行
- ✅ 自动备份
- ✅ 调用 Python 脚本处理 Protobuf

#### 3. **Rust 模块**

- **文件**: `src-tauri/src/modules/logout.rs`
- **用途**: 集成到 Tauri 应用中
- **功能**:
    - `clear_token()` - 清空数据库中的 Token
    - `clear_and_prepare_for_logout()` - 完整登出流程（关闭应用 + 清空 Token）

### 🔧 工作原理

1. **打开数据库** - 连接到 `state.vscdb`
2. **读取数据** - 获取 `jetskiStateSync.agentManagerInitState` 的值
3. **解码** - Base64 解码为二进制 Protobuf 数据
4. **移除 Field 6** - 删除包含 Token 的 Protobuf 字段
5. **编码** - Base64 重新编码
6. **写回** - 更新数据库

### 📍 数据库位置

| 系统        | 路径                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| **macOS**   | `~/Library/Application Support/Antigravity/User/globalStorage/state.vscdb` |
| **Windows** | `%APPDATA%\Antigravity\User\globalStorage\state.vscdb`                     |
| **Linux**   | `~/.config/Antigravity/User/globalStorage/state.vscdb`                     |

### 💻 使用示例

#### Python 脚本（交互式）

```bash
$ python3 clear_antigravity_token.py
🔧 Antigravity Token 清空工具

📁 数据库路径: /Users/user/Library/Application Support/Antigravity/User/globalStorage/state.vscdb

⚠️  确定要清空 Token 吗? (y/n): y
✅ 备份文件: /Users/user/Library/Application Support/Antigravity/User/globalStorage/state.vscdb.backup
📊 原始数据大小: 2048 字节
📊 清空后数据大小: 1024 字节
✅ Token 已成功清空！

✅ 操作完成！
   Antigravity 下次启动时将需要重新登录
```

#### Bash 脚本

```bash
$ ./clear_antigravity_token.sh
🔧 Antigravity Token 清空工具

📁 数据库路径: /Users/user/Library/Application Support/Antigravity/User/globalStorage/state.vscdb

⚠️  确定要清空 Token 吗? (y/n): y
✅ 备份文件: /Users/user/Library/Application Support/Antigravity/User/globalStorage/state.vscdb.backup
Token 已成功清空！

✅ 操作完成！
💡 提示: Antigravity 下次启动时将需要重新登录
```

### ⚠️ 注意事项

1. **关闭 Antigravity** - 建议运行脚本前关闭 Antigravity 应用
2. **备份** - 脚本自动备份原文件到 `state.vscdb.backup`
3. **权限** - 需要读写数据库文件的权限
4. **结果** - 清空后下次启动需要重新登录

### 🔄 与应用集成

在 Rust 中使用：

```rust
use crate::modules::logout;
use crate::modules::db;

// 清空指定数据库的 Token
let db_path = db::get_db_path()?;
logout::clear_token(&db_path)?;

// 或执行完整登出流程
logout::clear_and_prepare_for_logout(&account_id).await?;
```

### 🐛 故障排除

#### 问题：无法打开数据库

- **原因**: Antigravity 仍在运行，文件被占用
- **解决**: 关闭 Antigravity 应用后重试

#### 问题：未找到 state.vscdb

- **原因**: Antigravity 未安装或未运行过
- **解决**: 先启动 Antigravity，确保数据库存在

#### 问题：Protobuf 解析错误

- **原因**: 数据库格式不兼容
- **解决**: 检查 Antigravity 版本，使用备份文件恢复

### 📝 相关代码

- [db.rs](../src-tauri/src/modules/db.rs) - 数据库操作
- [protobuf.rs](../src-tauri/src/utils/protobuf.rs) - Protobuf 处理
- [account.rs](../src-tauri/src/modules/account.rs) - 账号管理
