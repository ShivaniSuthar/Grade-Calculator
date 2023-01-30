
import pandas as pd
import numpy as np
import os


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns
    a dictionary with the following structure:
    The keys are the general areas of the syllabus: lab, project,
    midterm, final, disc, checkpoint
    The values are lists that contain the assignment names of that type.
    For example the lab assignments all have names of the form labXX where XX
    is a zero-padded two digit number. See the doctests for more details.
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    all_columns = np.array(grades.columns)
    lab = []
    project = []
    midterm = []
    final = []
    disc = []
    checkpoint = []

    for c in all_columns:
        first_word = c.split(" ")[0]
        if (first_word[:3]=="lab"):
            lab.append(first_word)
        if (first_word[:7]=="project"):
            if (first_word[10:20]=="checkpoint"):
                checkpoint.append(first_word)
            else:
                project.append(first_word)
        if (first_word[:7]=="Midterm"):
            midterm.append(first_word)
        if (first_word[:5]=="Final"):
            final.append(first_word)
        if (first_word[:4]=="disc"):
            disc.append(first_word)

    lab = sorted(list(set(lab)), key=str)
    project = sorted(list(set(project)), key=str)
    midterm = sorted(list(set(midterm)), key=str)
    final = sorted(list(set(final)), key=str)
    disc = sorted(list(set(disc)), key=str)
    checkpoint = sorted(list(set(checkpoint)), key=str)

    output = {'lab': lab, 'project':project, 'midterm':midterm, 'final': final, 'disc': disc, 'checkpoint':checkpoint}
    return output

def projects_total(grades):
    '''
    projects_total takes in a DataFrame grades and returns the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''

    dict_main = get_assignment_names(grades)
    for col_name in dict_main['project']:
        grades[col_name] = grades[col_name].replace(np.nan, 0)

    mp = 0
    s = 0
    project_scores = []
    for project in dict_main['project']:
        mp += grades[project+' - Max Points']
        s += grades[project]
    return s/mp

def last_minute_submissions(grades):
    """
    last_minute_submissions takes in a DataFrame
    grades and returns a Series indexed by lab assignment that
    contains the number of submissions that were turned
    in on time by students that were marked "late".
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1, 10)])
    True
    >>> (out > 0).sum()
    8
    """

    all_columns = np.array(grades.columns)
    lab_late_col = []
    for col in all_columns:
        if "lab" in col:
            if "Lateness" in col:
                lab_late_col.append(col)

    lab_late_df = grades[lab_late_col]
    def convert_time(old):
        old = str(old)
        old = old.replace(" ","")
        dif_parts = old.split(":")

        hours = float(dif_parts[0])
        minutes = float(dif_parts[1])
        seconds = float(dif_parts[2])
        output = (hours) + minutes/60 + (seconds/60)/60
        return output

    cleaned_df = pd.DataFrame()
    threshold = 3

    for col_name in np.array(lab_late_df.columns):
        lab_name = col_name.split(" ")[0]
        temp_series = lab_late_df[col_name].apply(convert_time)
        cleaned_df[lab_name] = temp_series
    cleaned_df

    series_values = []
    for col_name in np.array(cleaned_df.columns):
        temp_col = np.array(cleaned_df[col_name])
        filtered_array = temp_col[(temp_col > 0) & (temp_col < threshold)]
        series_values.append(len(filtered_array))

    output = pd.Series(series_values)
    output = output.set_axis(list(cleaned_df.columns))
    return output

def lateness_penalty(col):
    """
    adjust_lateness takes in a Series containing
    how late a submission was processed
    and returns a Series of penalties according to the
    syllabus.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    def convert_time(old):
        old = str(old)
        old = old.replace(" ","")
        dif_parts = old.split(":")
        hours = float(dif_parts[0])
        minutes = float(dif_parts[1])
        seconds = float(dif_parts[2])
        output = (hours) + minutes/60 + (seconds/60)/60
        return output


    threshold = 3
    x = col.apply(convert_time)

    def hours_penalty(hours):
        if hours <= threshold:
            return 1
        elif hours <= 168:
            return 0.9
        elif hours <= 336:
            return 0.7
        return 0.4
    x = x.apply(hours_penalty)
    return x

def process_labs(grades):
    """
    process_labs takes in a DataFrame like grades and returns
    a DataFrame of processed lab scores. The output
      * has the same index as `grades`,
      * has one column for each lab assignment (e.g. `'lab01'`, `'lab02'`,..., `'lab09'`),
      * has values representing the final score for each lab assignment,
        adjusted for lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1, 10)]
    True
    >>> np.all((0.60 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    
    dict_main = get_assignment_names(grades)
    for col_name in dict_main['lab']:
        grades[col_name] = grades[col_name].replace(np.nan, 0)

    output = pd.DataFrame()

    for lab in dict_main['lab']:
        before_penalty=grades[lab]/grades[lab+' - Max Points']
        output[lab]=lateness_penalty(grades[lab+' - Lateness (H:M:S)'])*before_penalty
    return output


def lab_total(processed):
    """
    lab_total takes in DataFrame of processed assignments 
    and returns a Series containing the total lab grade for each
    student according to the syllabus.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    output = []
    out = processed.transpose()
    num_labs = len(list(out.index))
    for student in out:
        scores = out[student].sort_values(ascending=False)
        scores.reset_index(drop=True, inplace=True)
        scores = scores.drop(labels=[num_labs-1])
        avg = scores.sum() / (num_labs-1)
        output.append(avg)

    output = pd.Series(output)
    return output

def total_points(grades):
    """
    total_points takes in a DataFrame grades and returns a Series
    containing each student's course grade.
    Course grades should be proportions between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    for col_name in grades.columns:
        grades[col_name] = grades[col_name].replace(np.nan, 0)

    processed = process_labs(grades)
    lab_grades = lab_total(processed) * 0.20
    lab_grades

    final_scores = grades['Final']/grades['Final - Max Points'] * 0.30
    final_scores

    midterm_scores = grades['Midterm']/grades['Midterm - Max Points'] * 0.15
    midterm_scores

    di_col = []
    di_name = []
    for col in grades.columns:
        if "discussion" in col:
            if "Late" not in col:
                if "Max" not in col:
                    di_name.append(col)
                    di_col.append(col)
                else:
                    di_col.append(col)
    di_df = grades[di_col]
    new_di = pd.DataFrame()
    for di in di_name:
        new_di[di] = di_df[di] / di_df[di+' - Max Points']
    di_scores = (new_di.transpose().sum() / len(di_name)) * .025
    di_scores

    project_scores = projects_total(grades) * 0.30
    project_scores

    cp_df = pd.DataFrame()
    dict_main = get_assignment_names(grades)
    checkpoint_col = []
    for cp in grades[dict_main['checkpoint']]:
        cp_df[cp] = grades[cp] / grades[cp+' - Max Points']
    cp_scores = cp_df.mean(axis=1) * .025
    final = cp_scores + project_scores + lab_grades + di_scores + midterm_scores + final_scores
    return final

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.
    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def as_grade(percent):
        if percent >= 0.90:
            return "A"
        elif percent >= 0.80:
            return "B"
        elif percent >= 0.70:
            return "C"
        elif percent >= 0.60:
            return "D"
        return "F"
    return total.apply(as_grade)


def letter_proportions(total):
    """
    letter_proportions takes in the final course grades
    as above and outputs a Series that contains the
    proportion of the class that received each grade.
    :Example:
    >>> out = letter_proportions(pd.Series([0.99, 0.92, 0.89, 0.87, 0.82, 0.81, 0.80, 0.77, 0.77, 0.74]))
    >>> np.all(out.index == ['B', 'C', 'A'])
    True
    >>> out.sum() == 1.0
    True
    """

    return (final_grades(total).value_counts() / len(total)).sort_values(ascending=False)

def simulate_pval(grades, N):
    """
    simulate_pval takes in a DataFrame grades and
    a number of simulations N and returns the p-value
    for the hypothesis test described in the notebook.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 1000)
    >>> 0 <= out <= 0.1
    True
    """
    num_seniors = grades[grades["Level"]=="SR"].shape[0]
    just_grades = pd.DataFrame(total_points(grades))
    averages = []
    for i in range(N):
        random_sample = just_grades.sample(num_seniors)
        new_avg = random_sample[0].mean()
        averages.append(new_avg)

    senior_grades = grades.assign(total = total_points(grades))
    senior_grades = senior_grades[["Level", "total"]].groupby("Level").mean().reset_index()
    actual = senior_grades[senior_grades["Level"]=="SR"]["total"].iloc[0]
    observed_average = actual
    return (np.array(averages) >= observed_average).mean() / N


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades,
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    for col_name in grades.columns:
        grades[col_name] = grades[col_name].replace(np.nan, 0)
    dict_main = get_assignment_names(grades)

    lab_output = pd.DataFrame()
    for lab in dict_main['lab']:
        before_penalty=grades[lab]/grades[lab+' - Max Points']
        lab_output[lab]=(lateness_penalty(grades[lab+' - Lateness (H:M:S)'])*before_penalty) + np.random.normal(0, 0.02, 1)
        lab_output[lab]=np.clip(lab_output[lab], 0, 1)
    lab_grades = (lab_total(lab_output) * 0.20)
    lab_grades

    all_columns = np.array(grades.columns)
    relevant_columns = []
    project_names = []

    dict_main = get_assignment_names(grades)
    for col_name in dict_main['project']:
        grades[col_name] = grades[col_name].replace(np.nan, 0)

    for c in all_columns:
        first_word = c.split(" ")[0]
        if "Lateness" in c:
            first_word = "x"
        if (first_word[:7]=="project"):
            if (first_word[10:20]!="checkpoint"):
                relevant_columns.append(c)
                if len(c)==9:
                    project_names.append(c)
    rel_grades = grades[relevant_columns]
    rel_grades = rel_grades.fillna(0)
    relevent_col = np.array(rel_grades.columns)
    rel_grades = rel_grades.astype(float)
    project_score = []

    for project in project_names:
        score_total = [0,0]

        for col in relevent_col:
            if project in col:
                if "Max" in col:
                    score_total[1]+=rel_grades[col]
                else:
                    score_total[0]+=rel_grades[col]
        temp = (score_total[0]/score_total[1]) + np.random.normal(0, 0.02, 1)
        project_score.append(np.clip(temp, 0, 1))

    project_scores = (sum(project_score)/len(project_names)) * 0.30
    project_scores

    cp_df = pd.DataFrame()
    dict_main = get_assignment_names(grades)
    checkpoint_col = []
    for cp in grades[dict_main['checkpoint']]:
        cp_df[cp] = (grades[cp] / grades[cp+' - Max Points']) + np.random.normal(0, 0.02, 1)
        cp_df[cp] = np.clip(cp_df[cp], 0, 1)
    cp_scores = cp_df.mean(axis=1) * .025

    new_di = pd.DataFrame()
    for di in grades[dict_main['disc']]:
        new_di[di] = (grades[di] / grades[di+' - Max Points']) + np.random.normal(0, 0.02, 1)
        new_di[di] = np.clip(new_di[di], 0, 1)
    new_di
    di_scores = (new_di.transpose().sum() / len(dict_main['disc'])) * .025
    di_scores

    final_scores = (grades['Final']/grades['Final - Max Points'] + np.random.normal(0, 0.02, 1))
    final_scores = np.clip(final_scores, 0, 1) * 0.3

    midterm_scores = (grades['Midterm']/grades['Midterm - Max Points'] + np.random.normal(0, 0.02, 1))
    midterm_scores = np.clip(midterm_scores, 0, 1) * 0.15

    final = cp_scores + project_scores + lab_grades + di_scores + midterm_scores + final_scores
    return final
