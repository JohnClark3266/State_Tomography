#!/bin/bash
# 清空 Antigravity state.vscdb 中的 Token - Bash 版本
# 使用方法: ./clear_antigravity_token.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取操作系统
OS="$(uname -s)"
case "${OS}" in
    Darwin*)  OS_TYPE="macOS";;
    Linux*)   OS_TYPE="Linux";;
    CYGWIN*)  OS_TYPE="Windows";;
    *)        OS_TYPE="Unknown";;
esac

echo -e "${BLUE}🔧 Antigravity Token 清空工具${NC}\n"

# 获取数据库路径
if [ "$OS_TYPE" = "macOS" ]; then
    DB_PATH="$HOME/Library/Application Support/Antigravity/User/globalStorage/state.vscdb"
elif [ "$OS_TYPE" = "Linux" ]; then
    DB_PATH="$HOME/.config/Antigravity/User/globalStorage/state.vscdb"
elif [ "$OS_TYPE" = "Windows" ]; then
    # Windows 环境需要使用 Python 脚本
    echo -e "${YELLOW}⚠️  Windows 用户推荐使用 Python 脚本: python3 clear_antigravity_token.py${NC}"
    exit 1
else
    echo -e "${RED}❌ 不支持的操作系统${NC}"
    exit 1
fi

# 检查数据库文件是否存在
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}❌ 错误: 未找到 state.vscdb 文件${NC}"
    echo "   路径: $DB_PATH"
    echo "   请确保 Antigravity 已安装且至少运行过一次"
    exit 1
fi

echo -e "${BLUE}📁 数据库路径:${NC} $DB_PATH\n"

# 确认操作
read -p "$(echo -e ${YELLOW}⚠️  确定要清空 Token 吗? \(y/n\): ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ 操作已取消${NC}"
    exit 0
fi

# 备份原文件
BACKUP_PATH="${DB_PATH}.backup"
cp "$DB_PATH" "$BACKUP_PATH"
echo -e "${GREEN}✅ 备份文件:${NC} $BACKUP_PATH"

# 使用 sqlite3 和 base64 清空 Token
# 这需要使用 Python 来处理 Protobuf，所以调用 Python 脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/clear_antigravity_token.py"

if [ -f "$PYTHON_SCRIPT" ]; then
    python3 "$PYTHON_SCRIPT" --quiet
else
    echo -e "${YELLOW}⚠️  Python 脚本不存在: $PYTHON_SCRIPT${NC}"
    echo -e "${BLUE}💡 提示: 直接运行 Python 脚本:${NC}"
    echo "   python3 clear_antigravity_token.py"
    exit 1
fi

echo -e "\n${GREEN}✅ 操作完成！${NC}"
echo -e "${BLUE}💡 提示: Antigravity 下次启动时将需要重新登录${NC}"
