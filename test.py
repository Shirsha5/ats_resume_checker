# Create test_fix.py
from resume_parser import ResumeParser

parser = ResumeParser()
year = parser.extract_academic_year("(2021-26), Symbiosis Law School")
print(f"Year: {year}")  # Should print "Year: 5"
