"""
MathRAG PDF 文本加载器
功能：提取PDF全文文本，并保存为TXT文件
"""
import fitz  # PyMuPDF
import os


class PDFLoader:
    """PDF加载与文本提取器"""

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 文件不存在，请检查路径: {file_path}")
        self.file_path = file_path
        self.doc = fitz.open(file_path)

    def extract_full_text(self) -> str:
        """提取PDF全文"""
        full_text = ""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            if text.strip():
                full_text += text + "\n"
        return full_text

    def save_to_txt(self, output_path: str):
        """保存为TXT文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_text = self.extract_full_text()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✅ 文本提取成功！已保存至: {output_path}")
        print(f"📊 共提取 {len(self.doc)} 页，总字符数: {len(full_text)}")

    def close(self):
        self.doc.close()

# ---------- 直接运行这个文件测试 ----------
if __name__ == "__main__":
    import os
    from pathlib import Path

    # 自动定位项目根目录：从当前文件位置开始，向上找包含 data 文件夹的那个目录
    current_file = Path(__file__).resolve()  # C:\MathRAG\src\loader\pdf_loader.py
    project_root = current_file.parent.parent  # 先退到 src，再退一级就是 MathRAG

    # 但以防万一，如果退完还是不对，就继续往上找，直到找到 data 文件夹
    while not (project_root / "data").exists():
        if project_root.parent == project_root:  # 到根了还没找到，报错
            raise RuntimeError("找不到项目根目录（包含 data 文件夹的目录）")
        project_root = project_root.parent

    os.chdir(project_root)
    print(f"✅ 当前工作目录已切换至: {os.getcwd()}")

    raw_folder = "data/raw"

    # 2. 检查文件夹
    if not os.path.exists(raw_folder):
        print(f"❌ 文件夹不存在: {raw_folder}")
        print(f"请确保在 {os.getcwd()} 目录下存在 data/raw 文件夹")
        exit(1)

    # 3. 列出所有 PDF
    pdf_files = [f for f in os.listdir(raw_folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"❌ 在 {raw_folder} 文件夹里没有找到任何 PDF 文件！")
        print("请确认你把 PDF 放在了 data/raw/ 目录下。")
        exit(1)

    # 4. 如果有多个，选第一个；如果只有一个，直接用
    if len(pdf_files) == 1:
        pdf_name = pdf_files[0]
        print(f"📄 自动检测到 PDF: {pdf_name}")
    else:
        print("📄 检测到多个 PDF 文件，请选择：")
        for i, f in enumerate(pdf_files):
            print(f"  {i+1}. {f}")
        choice = input("请输入编号: ")
        try:
            pdf_name = pdf_files[int(choice)-1]
        except:
            print("输入无效，默认选第一个")
            pdf_name = pdf_files[0]

    # 5. 运行提取
    pdf_path = os.path.join(raw_folder, pdf_name)
    output_path = os.path.join("data", "processed", "full_text.txt")

    try:
        loader = PDFLoader(pdf_path)
        loader.save_to_txt(output_path)
        loader.close()
    except Exception as e:
        print(f"❌ 运行出错: {e}")