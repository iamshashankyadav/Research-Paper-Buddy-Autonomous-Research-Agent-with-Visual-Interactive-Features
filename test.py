from utils.pdf_loader import extract_text, split_into_sections
txt = extract_text("data/uploads/shashank_resume.pdf")
sections = split_into_sections(txt)
for s in sections[:3]:
    print(s["heading"], len(s["text"]))