from src.core.video.transcriber import VideoTranscript

if __name__ == "__main__":
    text = """Theo thông báo của UBND H.Hóc Môn, địa phương đang có nhu cầu tuyển dụng 340 viên chức ngành giáo dục - đào tạo trong năm 2024, trong đó gồm 309 giáo viên và 31 nhân viên. Cụ thể:

Bậc mầm non cần tuyển 53 người, gồm: 46 giáo viên và 7 nhân viên kế toán, tại các trường mầm non 2.9, 19.8, 23.11, Bà Điểm, Bé Ngoan, Bé Ngoan 1, Bé Ngoan 3, Bông Sen, Bông Sen 1, Hướng Dương, Nhị Xuân, Sơn Ca, Sơn Ca 3, Tân Hiệp, Tân Hòa, Tân Xuân, Xuân Thới Đông, Xuân Thới Thượng, Cúc Họa Mi.
Bậc tiểu học cần tuyển 109 người, gồm: 101 giáo viên và 8 nhân viên, tại các trường tiểu học Ấp Đình, Cầu Xáng, Dương Công Khi, Lý Chính Thắng 2, Mỹ Hòa, Mỹ Huề, Nam Kỳ Khởi Nghĩa, Ngã Ba Giồng, Nguyễn An Ninh, Nguyễn Thị Nuôi, Nhị Tân, Nhị Xuân, Tam Đông, Tam Đông 2, Tân Hiệp, Tân Xuân, Tây Bắc Lân, Thới Tam, Thới Thạnh, Trần Văn Danh, Trần Văn Mười, Võ Văn Thặng, Xuân Thới Thượng, Lê Văn Phiên.
Bậc trung học cơ sở cần tuyển 170 người, gồm: 155 giáo viên và 15 nhân viên, tại các trường THCS Bùi Văn Thủ, Đặng Công Bình, Đặng Thúc Vịnh, Đỗ Văn Dậy, Đông Thạnh, Lý Chính Thắng 1, Nguyễn An Khương, Nguyễn Hồng Đào, Nguyễn Văn Bứa, Phan Công Hớn, Tam Đông 1, Tân Xuân, Thị Trấn, Tô Ký, Trung Mỹ Tây 1, Xuân Thới Thượng.
Trung tâm Giáo dục nghề nghiệp - Giáo dục thường xuyên cần tuyển 8 người.
"""
    vt = VideoTranscript()
    vt.full_text = text
    vt.summary_subtitle()
    print("-==============")
