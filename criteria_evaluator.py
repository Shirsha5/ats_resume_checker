class CriteriaEvaluator:
    def __init__(self):
        self.long_term_criteria = [
            'academic_year_valid',
            'cgpa_sufficient', 
            'company_law_taken',
            'contract_law_taken',
            'legal_research_experience'
        ]

        self.short_term_criteria = [
            'academic_year_valid',
            'cgpa_sufficient'
        ]

        self.preference_criteria = [
            'moot_court_experience',
            'tier_firm_internship',
            'ma_moot_experience', 
            'publications',
            'faculty_recommendation',
            'legalogic_previous'
        ]

    def evaluate_academic_year(self, academic_year, course_type="5year"):
        """Evaluate if academic year meets requirements"""
        if academic_year is None:
            return False

        if course_type == "5year":
            return academic_year in [3, 4, 5]  # 3rd, 4th, or 5th year
        else:  # 3year course
            return academic_year in [2, 3]     # 2nd or 3rd year

    def evaluate_cgpa(self, cgpa, minimum=7.5):
        """Evaluate if CGPA meets minimum requirement"""
        if cgpa is None:
            return False
        return cgpa >= minimum

    def evaluate_long_term_eligibility(self, parsed_resume, course_type="5year"):
        """Evaluate eligibility for long-term internship"""
        score = 0
        max_score = 400  # 100 points each for 4 criteria
        criteria_met = {}

        # Academic Year (100 points)
        academic_valid = self.evaluate_academic_year(
            parsed_resume.get('academic_year'), course_type
        )
        criteria_met['academic_year'] = academic_valid
        if academic_valid:
            score += 100

        # CGPA (100 points)
        cgpa_valid = self.evaluate_cgpa(parsed_resume.get('cgpa'))
        criteria_met['cgpa'] = cgpa_valid
        if cgpa_valid:
            score += 100

        # Company Law (100 points)
        company_law = parsed_resume.get('company_law', False)
        criteria_met['company_law'] = company_law
        if company_law:
            score += 100

        # Contract Law (100 points)
        contract_law = parsed_resume.get('contract_law', False)
        criteria_met['contract_law'] = contract_law
        if contract_law:
            score += 100

        # Legal Research Experience (bonus for scoring)
        legal_research = parsed_resume.get('experience', {}).get('legal_research', False)
        criteria_met['legal_research'] = legal_research

        return {
            'eligible': score == max_score,  # Must meet ALL criteria
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'criteria_met': criteria_met,
            'category': 'long_term'
        }

    def evaluate_short_term_eligibility(self, parsed_resume, course_type="5year"):
        """Evaluate eligibility for short-term internship"""
        score = 0
        max_score = 200  # 100 points each for 2 criteria
        criteria_met = {}

        # Academic Year (100 points) - can be relaxed in exceptional circumstances
        academic_valid = self.evaluate_academic_year(
            parsed_resume.get('academic_year'), course_type
        )
        criteria_met['academic_year'] = academic_valid
        if academic_valid:
            score += 100

        # CGPA (100 points) - can be relaxed in exceptional circumstances  
        cgpa_valid = self.evaluate_cgpa(parsed_resume.get('cgpa'))
        criteria_met['cgpa'] = cgpa_valid
        if cgpa_valid:
            score += 100

        return {
            'eligible': score >= 100,  # At least one criteria must be met (first criteria mandatory for shortlisting)
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'criteria_met': criteria_met,
            'category': 'short_term'
        }

    def calculate_preference_score(self, parsed_resume):
        """Calculate additional preference score"""
        preference_score = 0
        preference_details = {}

        experience = parsed_resume.get('experience', {})

        # Moot Court Experience (50 points)
        if experience.get('moot_court', False):
            preference_score += 50
            preference_details['moot_court'] = 50
        else:
            preference_details['moot_court'] = 0

        # Check internships for tier 1/2 firms (40 points)
        if experience.get('tier_firm_internship', False):
            preference_score += 40
            preference_details['tier_firm'] = 40
        else:
            preference_details['tier_firm'] = 0

        # M&A Moot Court Experience (30 points)
        if experience.get('ma_moot_experience', False):
            preference_score += 30
            preference_details['ma_moot'] = 30
        else:
            preference_details['ma_moot'] = 0

        # Publications (30 points)
        if len(experience.get('publications', [])) > 0:
            preference_score += 30
            preference_details['publications'] = 30
        else:
            preference_details['publications'] = 0

        # Faculty Recommendation (20 points)
        if experience.get('faculty_recommendation', False):
            preference_score += 20
            preference_details['faculty_rec'] = 20
        else:
            preference_details['faculty_rec'] = 0

        # Previous LegaLogic Internship (25 points)
        if experience.get('legalogic_previous', False):
            preference_score += 25
            preference_details['legalogic_previous'] = 25
        else:
            preference_details['legalogic_previous'] = 0

        return preference_score, preference_details

    def classify_candidate(self, parsed_resume, course_type="5year", internship_type="long_term"):
        """Main classification function"""
        # Evaluate for long-term internship
        long_term_eval = self.evaluate_long_term_eligibility(parsed_resume, course_type)

        # Evaluate for short-term internship
        short_term_eval = self.evaluate_short_term_eligibility(parsed_resume, course_type)

        # Calculate preference score
        preference_score, preference_details = self.calculate_preference_score(parsed_resume)

        # Determine final classification
        classification = {
            'filename': parsed_resume.get('filename', 'unknown'),
            'cgpa': parsed_resume.get('cgpa'),
            'academic_year': parsed_resume.get('academic_year'),
            'long_term_evaluation': long_term_eval,
            'short_term_evaluation': short_term_eval,
            'preference_score': preference_score,
            'preference_details': preference_details,
            'experience_summary': parsed_resume.get('experience', {}),
            'final_category': 'others'  # Default category
        }

        # Classify based on evaluations and internship type preference
        if internship_type == "long_term":
            if long_term_eval['eligible']:
                classification['final_category'] = 'ma_team_match'
            elif short_term_eval['eligible']:
                classification['final_category'] = 'shortlisted'
        else:  # short_term
            if short_term_eval['eligible']:
                classification['final_category'] = 'shortlisted'
            # Long-term eligible candidates are also good for short-term
            elif long_term_eval['eligible']:
                classification['final_category'] = 'ma_team_match'

        # First criteria mandatory for shortlisting (academic year check)
        if not long_term_eval['criteria_met'].get('academic_year', False) and \
           not short_term_eval['criteria_met'].get('academic_year', False):
            # If academic year doesn't meet either criteria, check if other factors compensate
            if preference_score >= 70:  # High preference score might compensate
                classification['final_category'] = 'shortlisted'
                classification['special_consideration'] = 'High preference score despite academic year requirement'
            else:
                classification['final_category'] = 'others'

        return classification
