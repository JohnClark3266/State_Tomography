#!/usr/bin/env python3
"""
清空 Antigravity state.vscdb 中的登录 Token 脚本
跨平台支持: macOS, Windows, Linux
"""

import sqlite3
import base64
import os
import sys
import shutil
from pathlib import Path
from typing import Optional


def get_db_path() -> Optional[Path]:
    """获取 state.vscdb 的路径（跨平台）"""
    system = sys.platform
    
    if system == "darwin":  # macOS
        db_path = Path.home() / "Library/Application Support/Antigravity/User/globalStorage/state.vscdb"
    elif system == "win32":  # Windows
        appdata = os.getenv("APPDATA")
        if not appdata:
            print("❌ 错误: 无法获取 APPDATA 环境变量")
            return None
        db_path = Path(appdata) / "Antigravity/User/globalStorage/state.vscdb"
    else:  # Linux
        db_path = Path.home() / ".config/Antigravity/User/globalStorage/state.vscdb"
    
    return db_path if db_path.exists() else None


def remove_protobuf_field(data: bytes, field_num: int) -> bytes:
    """
    移除指定的 Protobuf 字段
    
    Args:
        data: Protobuf 二进制数据
        field_num: 要移除的字段号
    
    Returns:
        移除后的数据
    """
    result = bytearray()
    offset = 0
    
    def read_varint(data: bytes, offset: int) -> tuple[int, int]:
        """读取 Protobuf Varint"""
        result = 0
        shift = 0
        while True:
            if offset >= len(data):
                raise ValueError("数据不完整")
            byte = data[offset]
            result |= (byte & 0x7F) << shift
            offset += 1
            if byte & 0x80 == 0:
                break
            shift += 7
        return result, offset
    
    def skip_field(data: bytes, offset: int, wire_type: int) -> int:
        """跳过指定的 Protobuf 字段"""
        if wire_type == 0:  # Varint
            _, offset = read_varint(data, offset)
        elif wire_type == 1:  # 64-bit
            offset += 8
        elif wire_type == 2:  # Length-delimited
            length, offset = read_varint(data, offset)
            offset += length
        elif wire_type == 5:  # 32-bit
            offset += 4
        else:
            raise ValueError(f"未知的 wire_type: {wire_type}")
        return offset
    
    while offset < len(data):
        start_offset = offset
        tag, offset = read_varint(data, offset)
        wire_type = tag & 7
        current_field = tag >> 3
        
        if current_field == field_num:
            # 跳过这个字段
            offset = skip_field(data, offset, wire_type)
        else:
            # 保留其他字段
            next_offset = skip_field(data, offset, wire_type)
            result.extend(data[start_offset:next_offset])
            offset = next_offset
    
    return bytes(result)


def clear_token(db_path: Path, backup: bool = True) -> bool:
    """
    清空数据库中的 Token
    
    Args:
        db_path: state.vscdb 的路径
        backup: 是否备份原文件
    
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        # 1. 备份原文件
        if backup:
            backup_path = db_path.with_suffix(".vscdb.backup")
            shutil.copy2(db_path, backup_path)
            print(f"✅ 备份文件: {backup_path}")
        
        # 2. 打开数据库
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 3. 读取当前数据
        cursor.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            ("jetskiStateSync.agentManagerInitState",)
        )
        row = cursor.fetchone()
        
        if not row:
            print("⚠️  未找到 Token 数据，数据库可能已是清空状态")
            conn.close()
            return True
        
        current_data_b64 = row[0]
        
        # 4. Base64 解码
        blob = base64.b64decode(current_data_b64)
        print(f"📊 原始数据大小: {len(blob)} 字节")
        
        # 5. 移除 Field 6（Token 字段）
        clean_data = remove_protobuf_field(blob, 6)
        print(f"📊 清空后数据大小: {len(clean_data)} 字节")
        
        # 6. Base64 编码
        clean_b64 = base64.b64encode(clean_data).decode()
        
        # 7. 写回数据库
        cursor.execute(
            "UPDATE ItemTable SET value = ? WHERE key = ?",
            (clean_b64, "jetskiStateSync.agentManagerInitState")
        )
        conn.commit()
        conn.close()
        
        print("✅ Token 已成功清空！")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ 数据库错误: {e}")
        return False
    except ValueError as e:
        print(f"❌ Protobuf 解析错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False


def main():
    """主函数"""
    print("🔧 Antigravity Token 清空工具\n")
    
    # 获取数据库路径
    db_path = get_db_path()
    
    if not db_path:
        print("❌ 错误: 未找到 state.vscdb 文件")
        print("   请确保 Antigravity 已安装且至少运行过一次")
        return False
    
    print(f"📁 数据库路径: {db_path}\n")
    
    # 确认操作
    response = input("⚠️  确定要清空 Token 吗? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ 操作已取消")
        return False
    
    # 执行清空
    success = clear_token(db_path, backup=True)
    
    if success:
        print("\n✅ 操作完成！")
        print("   Antigravity 下次启动时将需要重新登录")
    else:
        print("\n❌ 操作失败，请检查数据库是否被占用")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
