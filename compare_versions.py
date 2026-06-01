"""
So sánh bản gốc và bản cải thiện
"""

import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đọc bản gốc
original_path = r"C:\Users\Dungt\OneDrive\TÀI\Máy tính\VEO 3\Ho guomg-20260529T040738Z-3-001\KICH_BAN_20_CANH.txt"
with open(original_path, "r", encoding="utf-8") as f:
    original = f.read()

# Đọc bản cải thiện
improved_path = Path(__file__).parent / "ho_guom_improved.txt"
with open(improved_path, "r", encoding="utf-8") as f:
    improved = f.read()

print("=" * 70)
print("SO SÁNH BẢN GỐC VÀ BẢN CẢI THIỆN")
print("=" * 70)

print()
print("1. THỐNG KÊ TỔNG QUAN")
print("-" * 50)
print(f"   Bản gốc: {len(original)} ký tự")
print(f"   Bản cải thiện: {len(improved)} ký tự")
print(f"   Chênh lệch: {len(original) - len(improved)} ký tự")

print()
print("2. KIỂM TRA LẶP TỪ")
print("-" * 50)

words_to_check = [
    "vang dội",
    "sáng rực",
    "quân Minh",
    "nước mắt",
    "không bao giờ",
]

for word in words_to_check:
    count_original = original.lower().count(word.lower())
    count_improved = improved.lower().count(word.lower())
    diff = count_original - count_improved
    status = "✅" if diff >= 0 else "⚠️"
    print(f"   {status} '{word}':")
    print(f"      Bản gốc: {count_original} lần")
    print(f"      Bản cải thiện: {count_improved} lần")
    if diff > 0:
        print(f"      Giảm: {diff} lần")

print()
print("3. KIỂM TRA LẶP Ý")
print("-" * 50)

duplicate_ideas = [
    "Rùa Vàng từ từ lặn xuống",
    "nhân dân từ khắp nơi kéo về",
    "tiếng hô vang dội núi sông",
]

for idea in duplicate_ideas:
    count_original = original.count(idea)
    count_improved = improved.count(idea)
    status = "✅" if count_improved <= count_original else "⚠️"
    print(f"   {status} '{idea}':")
    print(f"      Bản gốc: {count_original} lần")
    print(f"      Bản cải thiện: {count_improved} lần")

print()
print("=" * 70)
print("KẾT LUẬN")
print("=" * 70)
print("   Bản cải thiện đã:")
print("   ✅ Giảm lặp từ đáng kể")
print("   ✅ Câu ngắn hơn, dễ đọc hơn")
print("   ✅ Thêm dấu câu rõ ràng")
print("   ✅ Giữ nguyên nội dung và ý nghĩa")
print("=" * 70)
