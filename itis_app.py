import streamlit as st
import numpy as np
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="ITIS Calculator", layout="centered")

# ============================================================
# Session state for simple multi-page flow
# ============================================================
if "show_intro_page" not in st.session_state:
    st.session_state.show_intro_page = True
if "show_result_page" not in st.session_state:
    st.session_state.show_result_page = False
if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

# ============================================================
# Core model helpers
# ============================================================
def sigmoid_curve(x, A, n, d):
    return A / (1 + np.exp(n * (x - d)))

def calculate_n_from_vanish(d, vanish_day):
    if vanish_day <= d:
        return np.nan
    return np.log(99) / (vanish_day - d)

def make_linear(a1, a2, x1, x2):
    m = (a2 - a1) / (x2 - x1)
    b = a1 - m * x1
    return m, b

def compute_itis(days_since_iv, dose, cfg):
    m_A, b_A = make_linear(cfg["A1"], cfg["A2"], cfg["dose1"], cfg["dose2"])
    A = m_A * dose + b_A

    m_d, b_d = make_linear(cfg["d1"], cfg["d2"], cfg["dose1"], cfg["dose2"])
    d = m_d * dose + b_d

    n1 = calculate_n_from_vanish(cfg["d1"], cfg["vanish1"])
    n2 = calculate_n_from_vanish(cfg["d2"], cfg["vanish2"])
    m_n, b_n = make_linear(n1, n2, cfg["dose1"], cfg["dose2"])
    n = m_n * dose + b_n

    itis = sigmoid_curve(days_since_iv, A, n, d)
    return float(np.clip(itis, 0.0, 1.0))

def combine_itis(itis_values):
    """Cumulative ITIS = 1 - Π(1-ITIS)"""
    prod_term = 1.0
    for x in itis_values:
        prod_term *= (1.0 - x)
    return float(np.clip(1.0 - prod_term, 0.0, 1.0))

def group_into_courses(entries, window_days):
    """
    entries: list[(iv_date, dose)] sorted by iv_date

    Course rule:
    - include doses while (dose_date - course_start_date) <= window_days
    - cumulative course dose = sum of all doses in the course
    - course reference date = LAST dose date in that course

    Returns list[(course_last_date, course_total_dose)]
    """
    if not entries:
        return []

    entries = sorted(entries, key=lambda x: x[0])

    courses = []
    course_start = entries[0][0]
    course_last = entries[0][0]
    course_sum = entries[0][1]

    for dte, dse in entries[1:]:
        span = (dte - course_start).days
        if span <= window_days:
            course_sum += dse
            course_last = dte
        else:
            courses.append((course_last, course_sum))
            course_start = dte
            course_last = dte
            course_sum = dse

    courses.append((course_last, course_sum))
    return courses

# ============================================================
# Validation helpers
# ============================================================
def is_future_date(d):
    return d is not None and d > date.today()

def is_after_encounter(d, encounter_date):
    return d is not None and d > encounter_date

def clip_to_interval(value, low, high):
    return float(min(max(float(value), float(low)), float(high)))

def clip_course_total(course_total, upper):
    return min(float(course_total), float(upper))

def calculate_age_at_encounter(dob, encounter_date):
    if dob is None or encounter_date is None or dob > encounter_date:
        return np.nan
    return float((encounter_date - dob).days / 365.25)

def use_integer_input(cfg):
    keys = ["daily_min", "daily_max", "default_dose", "default_step"]
    return all(float(cfg[k]).is_integer() for k in keys)

# ============================================================
# Age-based dose adjustment helpers
# ============================================================
def apply_relative_age_adjustment(dose, age_years):
    """
    65-73: x1.5
    74-82: x2.0
    >82:   x2.5
    """
    if dose is None or np.isnan(dose):
        return dose
    if age_years is None or np.isnan(age_years):
        return float(dose)

    if 65 <= age_years <= 73:
        return float(dose) * 1.5
    elif 74 <= age_years <= 82:
        return float(dose) * 2.0
    elif age_years > 82:
        return float(dose) * 2.5
    return float(dose)

def apply_absolute_age_adjustment(dose, age_years):
    """
    65-73: +50
    74-82: +100
    >82:   +150
    """
    if dose is None or np.isnan(dose):
        return dose
    if age_years is None or np.isnan(age_years):
        return float(dose)

    if 65 <= age_years <= 73:
        return float(dose) + 50
    elif 74 <= age_years <= 82:
        return float(dose) + 100
    elif age_years > 82:
        return float(dose) + 150
    return float(dose)

# ============================================================
# Lymphocyte-based dose adjustment helpers
# Apply ONLY if lymphocyte tested == Yes and test date == encounter date
# and only AFTER age-based adjustment
# ============================================================
def should_apply_lymphocyte_adjustment(lymph_tested, lymph_test_date, encounter_date, lymph_value):
    if lymph_tested != "Yes":
        return False
    if lymph_test_date is None or encounter_date is None:
        return False
    if lymph_test_date != encounter_date:
        return False
    if lymph_value is None or np.isnan(lymph_value):
        return False
    return True

def apply_relative_lymphocyte_adjustment(dose, lymph_value):
    """
    >1.2                -> Apply algorithm (no extra change)
    >0.7 to <=1.2       -> Apply algorithm (no extra change)
    >0.3 to <=0.7       -> Increase dose by 50%
    <=0.3               -> Increase dose by 100%
    """
    if dose is None or np.isnan(dose):
        return dose
    if lymph_value is None or np.isnan(lymph_value):
        return float(dose)

    if lymph_value <= 0.3:
        return float(dose) * 2.0
    elif 0.3 < lymph_value <= 0.7:
        return float(dose) * 1.5
    else:
        return float(dose)

def apply_absolute_lymphocyte_adjustment(dose, lymph_value):
    """
    >1.2                -> Apply algorithm (no extra change)
    >0.7 to <=1.2       -> Apply algorithm (no extra change)
    >0.3 to <=0.7       -> Add 50 mg
    <=0.3               -> Add 100 mg
    """
    if dose is None or np.isnan(dose):
        return dose
    if lymph_value is None or np.isnan(lymph_value):
        return float(dose)

    if lymph_value <= 0.3:
        return float(dose) + 100.0
    elif 0.3 < lymph_value <= 0.7:
        return float(dose) + 50.0
    else:
        return float(dose)

# ============================================================
# CD19 helper for Rituximab only
# ============================================================
def apply_cd19_adjustment_for_rituximab(days_since_iv, iv_date, cd19_value, cd19_test_date):
    """
    Rules for Rituximab only:
    1. IF CD19 >0 and <10 -> apply algorithm (no change)
    2. IF CD19 >10 and interval between date of test and IV <=300 days -> RTX ITIS = 0
    3. IF CD19 = 0 and interval from IV <=30 days -> reset interval from IV to 0
    4. IF CD19 = 0 and interval from IV >30 days -> reset interval from IV to 30
    """
    if cd19_value is None or np.isnan(cd19_value):
        return days_since_iv, False

    if iv_date is None or cd19_test_date is None:
        return days_since_iv, False

    interval_test_from_iv = (cd19_test_date - iv_date).days

    if interval_test_from_iv < 0:
        return days_since_iv, False

    if cd19_value > 10 and interval_test_from_iv <= 300:
        return None, True

    if cd19_value == 0:
        if days_since_iv <= 30:
            return 0, False
        return 30, False

    return days_since_iv, False

# ============================================================
# Oral medication helpers
# ============================================================
def calculate_linear_score(dose, min_dose, max_dose, min_score, max_score):
    if dose <= min_dose:
        return float(min_score)
    elif dose >= max_dose:
        return float(max_score)
    else:
        return float(
            min_score + ((dose - min_dose) / (max_dose - min_dose)) * (max_score - min_score)
        )

# ============================================================
# Display helpers
# ============================================================
def date_display(d):
    return d.strftime("%d/%m/%Y")

# ============================================================
# Medication configs
# ============================================================
MEDS_IV = {
    "Methylprednisolone": {
        "dose1": 250, "dose2": 3000,
        "A1": 0.40, "A2": 0.80,
        "d1": 15.0, "d2": 21.0,
        "vanish1": 21.0, "vanish2": 30.0,
        "units": "mg",
        "course_window_days": 30,
        "course_cap_dose": 3000,
        "course_min_dose": 250,
        "max_n_doses": 20,
        "default_step": 50,
        "default_dose": 250,
        "age_adjustment_type": "relative",
        "lymphocyte_adjustment_type": "relative",
    },
    "Rituximab": {
        "dose1": 500, "dose2": 2000,
        "A1": 0.70, "A2": 0.85,
        "d1": 160.0, "d2": 200.0,
        "vanish1": 240.0, "vanish2": 300.0,
        "units": "mg",
        "course_window_days": 60,
        "course_cap_dose": 2000,
        "course_min_dose": 500,
        "max_n_doses": 20,
        "default_step": 100,
        "default_dose": 500,
        "age_adjustment_type": "relative",
        "lymphocyte_adjustment_type": "relative",
    },
    "Cyclophosphamide (IV)": {
        "dose1": 150, "dose2": 8000,
        "A1": 0.60, "A2": 0.90,
        "d1": 40.0, "d2": 80.0,
        "vanish1": 58.0, "vanish2": 110.0,
        "units": "mg",
        "course_window_days": 180,
        "course_cap_dose": 8000,
        "course_min_dose": 150,
        "max_n_doses": 30,
        "default_step": 50,
        "default_dose": 150,
        "age_adjustment_type": "relative",
        "lymphocyte_adjustment_type": "relative",
    },
}

ORAL_CYC = {
    "dose1": 75, "dose2": 25000,
    "A1": 0.60, "A2": 0.90,
    "d1": 70.0, "d2": 90.0,
    "vanish1": 90.0, "vanish2": 115.0,
    "course_min": 75,
    "course_max": 25000,
    "daily_dose_units": "mg/day",
    "daily_min": 0,
    "daily_max": 1000,
    "max_n_courses": 20,
    "age_adjustment_type": "relative",
    "lymphocyte_adjustment_type": "relative",
}

AZATHIOPRINE = {
    "dose1": 25,
    "dose2": 250,
    "A1": 0.15,
    "A2": 0.60,
    "d1": 8.0,
    "d2": 10.0,
    "vanish1": 10.0,
    "vanish2": 14.0,
    "min_score": 0.15,
    "max_score": 0.60,
    "daily_dose_units": "mg/day",
    "daily_min": 25,
    "daily_max": 250,
    "max_n_courses": 20,
    "default_dose": 25,
    "default_step": 25,
    "age_adjustment_type": "absolute",
    "lymphocyte_adjustment_type": "absolute",
}

MYCOPHENOLATE_MOFETIL = {
    "dose1": 125,
    "dose2": 4000,
    "A1": 0.25,
    "A2": 0.75,
    "d1": 5.0,
    "d2": 7.0,
    "vanish1": 7.0,
    "vanish2": 9.0,
    "min_score": 0.25,
    "max_score": 0.75,
    "daily_dose_units": "mg/day",
    "daily_min": 125,
    "daily_max": 4000,
    "max_n_courses": 20,
    "default_dose": 125,
    "default_step": 125,
    "age_adjustment_type": "relative",
    "lymphocyte_adjustment_type": "relative",
}

METHOTREXATE = {
    "dose1": 0.35,
    "dose2": 5.0,
    "A1": 0.10,
    "A2": 0.55,
    "d1": 8.0,
    "d2": 10.0,
    "vanish1": 10.0,
    "vanish2": 14.0,
    "min_score": 0.10,
    "max_score": 0.55,
    "daily_dose_units": "mg",
    "daily_min": 0.35,
    "daily_max": 5.0,
    "max_n_courses": 20,
    "default_dose": 0.35,
    "default_step": 0.05,
    "age_adjustment_type": "absolute",
    "lymphocyte_adjustment_type": "none",
}

TACROLIMUS = {
    "dose1": 1.0,
    "dose2": 6.0,
    "A1": 0.50,
    "A2": 0.75,
    "d1": 12.0,
    "d2": 12.0,
    "vanish1": 13.0,
    "vanish2": 14.0,
    "min_score": 0.50,
    "max_score": 0.75,
    "daily_dose_units": "mg",
    "daily_min": 1.0,
    "daily_max": 6.0,
    "max_n_courses": 20,
    "default_dose": 1.0,
    "default_step": 0.5,
    "age_adjustment_type": "none",
    "lymphocyte_adjustment_type": "none",
}

PREDNISOLONE = {
    "dose_categories": [
        "< 5 mg/day",
        "5 - 10 mg/day",
        "11 - 20 mg/day",
        "> 20 mg/day",
    ],
    "A_map": {
        "< 5 mg/day": 0.10,
        "5 - 10 mg/day": 0.25,
        "11 - 20 mg/day": 0.45,
        "> 20 mg/day": 0.75,
    },
    "d_map": {
        "< 5 mg/day": 5,
        "5 - 10 mg/day": 6,
        "11 - 20 mg/day": 7,
        "> 20 mg/day": 7,
    },
    "vanish_map": {
        "< 5 mg/day": 7,
        "5 - 10 mg/day": 9,
        "11 - 20 mg/day": 12,
        "> 20 mg/day": 15,
    },
    "max_n_courses": 20,
    "default_category": "5 - 10 mg/day",
}

# ============================================================
# Reusable oral medication section
# ============================================================
def render_decay_oral_medication_section(
    med_name,
    cfg,
    received_key,
    n_courses_key,
    start_prefix,
    stop_prefix,
    stop_disabled_prefix,
    not_stopped_prefix,
    dose_prefix,
):
    st.subheader(med_name)

    med_received = st.radio(
        f"Received {med_name}?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key=received_key,
    )

    if med_received == "Yes":
        st.info(
            "If the medication has not been stopped yet, set the Stop date to the Encounter/Current date "
            "to estimate the current approximate score."
        )

        st.number_input(
            f"How many {med_name.lower()} courses were given?",
            min_value=1,
            max_value=int(cfg["max_n_courses"]),
            value=int(st.session_state.get(n_courses_key, 1)),
            step=1,
            key=n_courses_key,
        )

        for i in range(int(st.session_state[n_courses_key])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state.get(
                        f"{start_prefix}_{i}",
                        st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today()
                    ),
                    max_value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                    format="DD/MM/YYYY",
                    key=f"{start_prefix}_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=bool(st.session_state.get(f"{not_stopped_prefix}_{i}", False)),
                key=f"{not_stopped_prefix}_{i}",
            )

            with c2:
                if st.session_state[f"{not_stopped_prefix}_{i}"]:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"{stop_disabled_prefix}_{i}",
                    )
                else:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state.get(
                            f"{stop_prefix}_{i}",
                            st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today()
                        ),
                        max_value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                        format="DD/MM/YYYY",
                        key=f"{stop_prefix}_{i}",
                    )

            if use_integer_input(cfg):
                st.number_input(
                    f"Dose #{i+1} ({cfg['daily_dose_units']})",
                    min_value=int(cfg["daily_min"]),
                    value=int(st.session_state.get(f"{dose_prefix}_{i}", cfg["default_dose"])),
                    step=int(cfg["default_step"]),
                    format="%d",
                    key=f"{dose_prefix}_{i}",
                )
            else:
                st.number_input(
                    f"Dose #{i+1} ({cfg['daily_dose_units']})",
                    min_value=float(cfg["daily_min"]),
                    value=float(st.session_state.get(f"{dose_prefix}_{i}", cfg["default_dose"])),
                    step=float(cfg["default_step"]),
                    format="%.2f",
                    key=f"{dose_prefix}_{i}",
                )

            if i < int(st.session_state[n_courses_key]) - 1:
                st.divider()
    else:
        st.caption("Not included (not received).")

def render_prednisolone_section():
    st.subheader("Prednisolone")

    prd_received = st.radio(
        "Received Prednisolone?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key="prd_received",
    )

    if prd_received == "Yes":
        st.info(
            "If the medication has not been stopped yet, set the Stop date to the Encounter/Current date "
            "to estimate the current approximate score."
        )

        st.number_input(
            "How many prednisolone courses were given?",
            min_value=1,
            max_value=int(PREDNISOLONE["max_n_courses"]),
            value=int(st.session_state.get("prd_n_courses", 1)),
            step=1,
            key="prd_n_courses",
        )

        for i in range(int(st.session_state["prd_n_courses"])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state.get(
                        f"prd_start_{i}",
                        st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today()
                    ),
                    max_value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                    format="DD/MM/YYYY",
                    key=f"prd_start_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=bool(st.session_state.get(f"prd_not_stopped_{i}", False)),
                key=f"prd_not_stopped_{i}",
            )

            with c2:
                if st.session_state[f"prd_not_stopped_{i}"]:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"prd_stop_disabled_{i}",
                    )
                else:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state.get(
                            f"prd_stop_{i}",
                            st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today()
                        ),
                        max_value=st.session_state["global_encounter_date"] if "global_encounter_date" in st.session_state else date.today(),
                        format="DD/MM/YYYY",
                        key=f"prd_stop_{i}",
                    )

            st.selectbox(
                f"Dose category #{i+1}",
                options=PREDNISOLONE["dose_categories"],
                index=PREDNISOLONE["dose_categories"].index(
                    st.session_state.get(f"prd_dose_cat_{i}", PREDNISOLONE["default_category"])
                ),
                key=f"prd_dose_cat_{i}",
            )

            if i < int(st.session_state["prd_n_courses"]) - 1:
                st.divider()
    else:
        st.caption("Not included (not received).")

# ============================================================
# Calculation
# ============================================================
def calculate_all_results():
    encounter_date = st.session_state["global_encounter_date"]
    dob = st.session_state.get("date_of_birth")
    age_at_encounter = calculate_age_at_encounter(dob, encounter_date)

    lymph_tested = st.session_state.get("lymphocyte_tested", "No")
    lymph_test_date = st.session_state.get("lymphocyte_test_date", None)
    lymphocyte_count = st.session_state.get("lymphocyte_count", None)
    try:
        lymphocyte_count = float(lymphocyte_count) if lymphocyte_count is not None else np.nan
    except Exception:
        lymphocyte_count = np.nan

    apply_lymph = should_apply_lymphocyte_adjustment(
        lymph_tested=lymph_tested,
        lymph_test_date=lymph_test_date,
        encounter_date=encounter_date,
        lymph_value=lymphocyte_count,
    )

    cd19_tested = st.session_state.get("cd19_tested", "No")
    cd19_test_date = st.session_state.get("cd19_test_date", None)
    cd19_value = st.session_state.get("cd19_value", None)
    try:
        cd19_value = float(cd19_value) if cd19_value is not None else np.nan
    except Exception:
        cd19_value = np.nan

    any_errors = False
    overall_components = []
    summary_lines = []

    # --------------------------------------------------------
    # IV meds
    # --------------------------------------------------------
    for med_name, cfg in MEDS_IV.items():
        received = st.session_state.get(f"{med_name}_received", "No")

        if received == "No":
            continue

        n_doses = int(st.session_state.get(f"{med_name}_n_doses", 1))
        entries = []
        invalid_found = False
        med_entered_doses = []

        for i in range(n_doses):
            raw_dose = int(st.session_state.get(f"{med_name}_dose_{i}", cfg["default_dose"]))
            iv_date = st.session_state.get(f"{med_name}_date_{i}", encounter_date)

            if is_future_date(iv_date) or is_after_encounter(iv_date, encounter_date):
                any_errors = True
                invalid_found = True

            if raw_dose < 0:
                any_errors = True
                invalid_found = True

            adjusted_dose = float(raw_dose)

            if cfg.get("age_adjustment_type") == "relative":
                adjusted_dose = apply_relative_age_adjustment(adjusted_dose, age_at_encounter)
            elif cfg.get("age_adjustment_type") == "absolute":
                adjusted_dose = apply_absolute_age_adjustment(adjusted_dose, age_at_encounter)

            if apply_lymph:
                if cfg.get("lymphocyte_adjustment_type") == "relative":
                    adjusted_dose = apply_relative_lymphocyte_adjustment(adjusted_dose, lymphocyte_count)
                elif cfg.get("lymphocyte_adjustment_type") == "absolute":
                    adjusted_dose = apply_absolute_lymphocyte_adjustment(adjusted_dose, lymphocyte_count)

            adjusted_dose = clip_to_interval(adjusted_dose, cfg["dose1"], cfg["dose2"])

            entries.append((iv_date, adjusted_dose))
            med_entered_doses.append(f"{date_display(iv_date)}: {raw_dose} mg")

        if invalid_found:
            summary_lines.append(f"- {med_name}: excluded due to invalid input(s).")
            continue

        entries.sort(key=lambda x: x[0])
        courses = group_into_courses(entries, window_days=int(cfg["course_window_days"]))

        med_course_itises = []
        for course_last_date, course_sum_dose in courses:
            if is_future_date(course_last_date) or is_after_encounter(course_last_date, encounter_date):
                any_errors = True
                med_course_itises = []
                break

            course_dose_used = clip_course_total(course_sum_dose, cfg["course_cap_dose"])

            if course_dose_used < float(cfg["course_min_dose"]):
                any_errors = True
                continue

            days_since = (encounter_date - course_last_date).days
            if days_since < 0:
                days_since = 0

            if med_name == "Rituximab" and cd19_tested == "Yes":
                days_since, force_itis_zero = apply_cd19_adjustment_for_rituximab(
                    days_since_iv=days_since,
                    iv_date=course_last_date,
                    cd19_value=cd19_value,
                    cd19_test_date=cd19_test_date,
                )
                if force_itis_zero:
                    med_course_itises.append(0.0)
                    continue

            med_course_itises.append(compute_itis(days_since, course_dose_used, cfg))

        if med_course_itises:
            overall_components.append(combine_itis(med_course_itises))
            summary_lines.append(f"- {med_name}: " + "; ".join(med_entered_doses))
        else:
            summary_lines.append(f"- {med_name}: no valid course included.")

    # --------------------------------------------------------
    # Oral Cyclophosphamide
    # --------------------------------------------------------
    if st.session_state.get("oral_cyc_received", "No") == "Yes":
        n_oral_courses = int(st.session_state.get("oral_cyc_n_courses", 1))
        oral_course_itises = []
        oral_summary = []

        for i in range(n_oral_courses):
            oral_start = st.session_state.get(f"oral_cyc_start_{i}", encounter_date)
            not_stopped = st.session_state.get(f"oral_cyc_not_stopped_{i}", False)
            oral_stop = encounter_date if not_stopped else st.session_state.get(f"oral_cyc_stop_{i}", encounter_date)
            raw_daily_dose = float(st.session_state.get(f"oral_cyc_daily_dose_{i}", 75))

            oral_invalid = False
            if is_future_date(oral_start) or is_after_encounter(oral_start, encounter_date):
                any_errors = True
                oral_invalid = True
            if is_future_date(oral_stop) or is_after_encounter(oral_stop, encounter_date):
                any_errors = True
                oral_invalid = True
            if oral_stop < oral_start:
                any_errors = True
                oral_invalid = True
            if raw_daily_dose < ORAL_CYC["daily_min"] or raw_daily_dose > ORAL_CYC["daily_max"]:
                any_errors = True
                oral_invalid = True

            if not oral_invalid:
                adjusted_daily_dose = float(raw_daily_dose)

                if ORAL_CYC.get("age_adjustment_type") == "relative":
                    adjusted_daily_dose = apply_relative_age_adjustment(adjusted_daily_dose, age_at_encounter)
                elif ORAL_CYC.get("age_adjustment_type") == "absolute":
                    adjusted_daily_dose = apply_absolute_age_adjustment(adjusted_daily_dose, age_at_encounter)

                if apply_lymph:
                    if ORAL_CYC.get("lymphocyte_adjustment_type") == "relative":
                        adjusted_daily_dose = apply_relative_lymphocyte_adjustment(adjusted_daily_dose, lymphocyte_count)
                    elif ORAL_CYC.get("lymphocyte_adjustment_type") == "absolute":
                        adjusted_daily_dose = apply_absolute_lymphocyte_adjustment(adjusted_daily_dose, lymphocyte_count)

                adjusted_daily_dose = clip_to_interval(
                    adjusted_daily_dose,
                    ORAL_CYC["daily_min"],
                    ORAL_CYC["daily_max"]
                )

                effective_stop = min(oral_stop, encounter_date)
                days_on_drug = (effective_stop - oral_start).days
                course_total = float(days_on_drug) * float(adjusted_daily_dose)

                if course_total < ORAL_CYC["course_min"]:
                    any_errors = True
                    oral_summary.append(
                        f"course #{i+1}: excluded (entered {raw_daily_dose:.2f} mg/day, {date_display(oral_start)} to {date_display(oral_stop)})"
                    )
                else:
                    course_total_used = clip_course_total(course_total, ORAL_CYC["course_max"])
                    interval_since_stop = (encounter_date - oral_stop).days
                    if interval_since_stop < 0:
                        interval_since_stop = 0

                    oral_course_itises.append(
                        compute_itis(interval_since_stop, course_total_used, ORAL_CYC)
                    )
                    oral_summary.append(
                        f"course #{i+1}: {date_display(oral_start)} to {date_display(oral_stop)}, dose {raw_daily_dose:.2f} mg/day"
                    )
            else:
                oral_summary.append(f"course #{i+1}: excluded due to invalid input(s).")

        if oral_course_itises:
            overall_components.append(combine_itis(oral_course_itises))
            summary_lines.append("- Cyclophosphamide (Oral): " + "; ".join(oral_summary))
        else:
            summary_lines.append("- Cyclophosphamide (Oral): no valid course included.")

    # --------------------------------------------------------
    # Generic decay oral medication calculator
    # --------------------------------------------------------
    def calculate_decay_oral_medication(
        med_name,
        cfg,
        received_key,
        n_courses_key,
        start_prefix,
        stop_prefix,
        not_stopped_prefix,
        dose_prefix,
    ):
        nonlocal any_errors, overall_components, summary_lines, encounter_date, age_at_encounter, apply_lymph, lymphocyte_count

        if st.session_state.get(received_key, "No") != "Yes":
            return

        n_courses = int(st.session_state.get(n_courses_key, 1))
        med_course_itises = []
        med_summary = []

        for i in range(n_courses):
            med_start = st.session_state.get(f"{start_prefix}_{i}", encounter_date)
            med_not_stopped = st.session_state.get(f"{not_stopped_prefix}_{i}", False)
            med_stop = encounter_date if med_not_stopped else st.session_state.get(f"{stop_prefix}_{i}", encounter_date)
            raw_dose = float(st.session_state.get(f"{dose_prefix}_{i}", cfg["default_dose"]))

            med_invalid = False
            if is_future_date(med_start) or is_after_encounter(med_start, encounter_date):
                any_errors = True
                med_invalid = True
            if is_future_date(med_stop) or is_after_encounter(med_stop, encounter_date):
                any_errors = True
                med_invalid = True
            if med_stop < med_start:
                any_errors = True
                med_invalid = True
            if raw_dose < cfg["daily_min"] or raw_dose > cfg["daily_max"]:
                any_errors = True
                med_invalid = True

            if not med_invalid:
                adjusted_dose = float(raw_dose)

                if cfg.get("age_adjustment_type") == "relative":
                    adjusted_dose = apply_relative_age_adjustment(adjusted_dose, age_at_encounter)
                elif cfg.get("age_adjustment_type") == "absolute":
                    adjusted_dose = apply_absolute_age_adjustment(adjusted_dose, age_at_encounter)

                if apply_lymph:
                    if cfg.get("lymphocyte_adjustment_type") == "relative":
                        adjusted_dose = apply_relative_lymphocyte_adjustment(adjusted_dose, lymphocyte_count)
                    elif cfg.get("lymphocyte_adjustment_type") == "absolute":
                        adjusted_dose = apply_absolute_lymphocyte_adjustment(adjusted_dose, lymphocyte_count)

                adjusted_dose = clip_to_interval(
                    adjusted_dose,
                    cfg["daily_min"],
                    cfg["daily_max"]
                )

                if encounter_date <= med_stop:
                    med_itis = calculate_linear_score(
                        adjusted_dose,
                        cfg["dose1"],
                        cfg["dose2"],
                        cfg["min_score"],
                        cfg["max_score"],
                    )
                else:
                    interval_since_stop = (encounter_date - med_stop).days
                    if interval_since_stop < 0:
                        interval_since_stop = 0
                    med_itis = compute_itis(interval_since_stop, adjusted_dose, cfg)

                med_course_itises.append(med_itis)
                med_summary.append(
                    f"course #{i+1}: {date_display(med_start)} to {date_display(med_stop)}, dose {raw_dose:.2f} {cfg['daily_dose_units']}"
                )
            else:
                med_summary.append(f"course #{i+1}: excluded due to invalid input(s).")

        if med_course_itises:
            overall_components.append(combine_itis(med_course_itises))
            summary_lines.append(f"- {med_name}: " + "; ".join(med_summary))
        else:
            summary_lines.append(f"- {med_name}: no valid course included.")

    calculate_decay_oral_medication(
        med_name="Azathioprine",
        cfg=AZATHIOPRINE,
        received_key="aza_received",
        n_courses_key="aza_n_courses",
        start_prefix="aza_start",
        stop_prefix="aza_stop",
        not_stopped_prefix="aza_not_stopped",
        dose_prefix="aza_daily_dose",
    )

    calculate_decay_oral_medication(
        med_name="Mycophenolate mofetil",
        cfg=MYCOPHENOLATE_MOFETIL,
        received_key="mmf_received",
        n_courses_key="mmf_n_courses",
        start_prefix="mmf_start",
        stop_prefix="mmf_stop",
        not_stopped_prefix="mmf_not_stopped",
        dose_prefix="mmf_daily_dose",
    )

    calculate_decay_oral_medication(
        med_name="Methotrexate",
        cfg=METHOTREXATE,
        received_key="meth_received",
        n_courses_key="meth_n_courses",
        start_prefix="meth_start",
        stop_prefix="meth_stop",
        not_stopped_prefix="meth_not_stopped",
        dose_prefix="meth_dose",
    )

    calculate_decay_oral_medication(
        med_name="Tacrolimus",
        cfg=TACROLIMUS,
        received_key="tac_received",
        n_courses_key="tac_n_courses",
        start_prefix="tac_start",
        stop_prefix="tac_stop",
        not_stopped_prefix="tac_not_stopped",
        dose_prefix="tac_dose",
    )

    # --------------------------------------------------------
    # Avacopan
    # --------------------------------------------------------
    if st.session_state.get("avacopan_received", "No") == "Yes":
        overall_components.append(0.50)
        summary_lines.append("- Avacopan: fixed dose 60 mg, ITIS = 0.50")

    # --------------------------------------------------------
    # Prednisolone
    # --------------------------------------------------------
    if st.session_state.get("prd_received", "No") == "Yes":
        n_prd_courses = int(st.session_state.get("prd_n_courses", 1))
        prd_course_itises = []
        prd_summary = []

        for i in range(n_prd_courses):
            prd_start = st.session_state.get(f"prd_start_{i}", encounter_date)
            prd_not_stopped = st.session_state.get(f"prd_not_stopped_{i}", False)
            prd_stop = encounter_date if prd_not_stopped else st.session_state.get(f"prd_stop_{i}", encounter_date)
            dose_cat = st.session_state.get(f"prd_dose_cat_{i}", PREDNISOLONE["default_category"])

            prd_invalid = False
            if is_future_date(prd_start) or is_after_encounter(prd_start, encounter_date):
                any_errors = True
                prd_invalid = True
            if is_future_date(prd_stop) or is_after_encounter(prd_stop, encounter_date):
                any_errors = True
                prd_invalid = True
            if prd_stop < prd_start:
                any_errors = True
                prd_invalid = True
            if dose_cat not in PREDNISOLONE["A_map"]:
                any_errors = True
                prd_invalid = True

            if not prd_invalid:
                if encounter_date <= prd_stop:
                    prd_itis = float(PREDNISOLONE["A_map"][dose_cat])
                else:
                    interval_since_stop = (encounter_date - prd_stop).days
                    if interval_since_stop < 0:
                        prd_itis = 0.0
                    else:
                        A = float(PREDNISOLONE["A_map"][dose_cat])
                        d = float(PREDNISOLONE["d_map"][dose_cat])
                        vanish = float(PREDNISOLONE["vanish_map"][dose_cat])
                        n = calculate_n_from_vanish(d, vanish)
                        prd_itis = float(np.clip(sigmoid_curve(interval_since_stop, A, n, d), 0.0, 1.0))

                prd_course_itises.append(prd_itis)
                prd_summary.append(
                    f"course #{i+1}: {date_display(prd_start)} to {date_display(prd_stop)}, dose {dose_cat}"
                )
            else:
                prd_summary.append(f"course #{i+1}: excluded due to invalid input(s).")

        if prd_course_itises:
            overall_components.append(combine_itis(prd_course_itises))
            summary_lines.append("- Prednisolone: " + "; ".join(prd_summary))
        else:
            summary_lines.append("- Prednisolone: no valid course included.")

    cumulative_itis = combine_itis(overall_components)

    return {
        "encounter_date": encounter_date,
        "date_of_birth": dob,
        "age_at_encounter": age_at_encounter,
        "lymphocyte_tested": lymph_tested,
        "lymphocyte_test_date": lymph_test_date,
        "lymphocyte_count": lymphocyte_count,
        "lymphocyte_applied": apply_lymph,
        "cd19_tested": cd19_tested,
        "cd19_test_date": cd19_test_date,
        "cd19_value": cd19_value,
        "cumulative_itis": cumulative_itis,
        "any_errors": any_errors,
        "summary_lines": summary_lines,
    }

# ============================================================
# Introduction page
# ============================================================
if st.session_state.show_intro_page:
    st.title("Immunosuppressive Therapy Intensity Score (ITIS)")

    st.write(
        "The immunosuppressive therapy intensity score (ITIS) is derived as part of the "
        "[PARADISE](https://paradise-project.eu/) project."
    )
    st.write("This tool provides an approximate score of the degree of immunosuppression.")
    st.write(
        "Any values closer to one indicate suppressed immunity, whereas any score closer "
        "to zero indicates normal immune function."
    )
    st.write(
        "This tool can be used by clinicians, researchers, or patients. "
        "It represents a patient's overall treatment status at a specific time point."
    )

    st.subheader("Information required to calculate ITIS")
    st.write("• Date of birth")
    st.write("• Encounter / current date")
    st.write("• Lymphocyte count (optional)")
    st.write("• CD19 count (optional)")
    st.write("• Medication")
    st.write("• Date of IV (for IV medications)")
    st.write("• Start date and stop date(s) for oral medications")
    st.write("• Units must be entered as mg")
    st.write("• Medication dose (enter the daily dose for oral medications)")
    st.divider()

    if st.button("Continue", type="primary"):
        st.session_state.show_intro_page = False
        st.rerun()

# ============================================================
# Result page
# ============================================================
elif st.session_state.show_result_page and st.session_state.result_payload is not None:
    result = st.session_state.result_payload

    st.title("Estimated Cumulative ITIS Result")
    st.caption(f"Encounter / Current Date: {date_display(result['encounter_date'])}")

    st.metric("Estimated Cumulative ITIS", f"≈ {result['cumulative_itis']:.2f}")

    st.subheader("Summary")
    if result.get("date_of_birth") is not None:
        st.write(f"**Date of birth:** {date_display(result['date_of_birth'])}")
    if result.get("age_at_encounter") is not None and not np.isnan(result["age_at_encounter"]):
        st.write(f"**Age at encounter:** {result['age_at_encounter']:.1f} years")

    st.write(f"**Lymphocyte count tested:** {result.get('lymphocyte_tested', 'No')}")
    if result.get("lymphocyte_tested") == "Yes":
        if result.get("lymphocyte_test_date") is not None:
            st.write(f"**Lymphocyte test date:** {date_display(result['lymphocyte_test_date'])}")
        if result.get("lymphocyte_count") is not None and not np.isnan(result["lymphocyte_count"]):
            st.write(f"**Lymphocyte count:** {result['lymphocyte_count']:.2f} ×10⁹/L")

    st.write(f"**CD19 tested:** {result.get('cd19_tested', 'No')}")
    if result.get("cd19_tested") == "Yes":
        if result.get("cd19_test_date") is not None:
            st.write(f"**CD19 test date:** {date_display(result['cd19_test_date'])}")
        if result.get("cd19_value") is not None and not np.isnan(result["cd19_value"]):
            st.write(f"**CD19 value:** {result['cd19_value']:.2f}")

    st.subheader("Summary of Entered Medications")
    if result["summary_lines"]:
        for line in result["summary_lines"]:
            st.write(line)
    else:
        st.write("No medications were entered.")

    if result["lymphocyte_tested"] == "Yes" and not result["lymphocyte_applied"]:
        st.info("Lymphocyte-based dose adjustment was not applied because the lymphocyte test date did not match the encounter date, or the lymphocyte count was invalid or missing.")

    if result["any_errors"]:
        st.warning("One or more inputs were invalid. Some medications or courses may have been excluded.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Back to entry form"):
            st.session_state.show_result_page = False
            st.rerun()

    with c2:
        if st.button("Back to introduction"):
            st.session_state.show_result_page = False
            st.session_state.show_intro_page = True
            st.rerun()

# ============================================================
# Entry page
# ============================================================
else:
    st.title("Immunosuppressive Therapy Intensity Score (ITIS)")

    st.subheader("Patient details")
    st.caption("Please enter/select dates in DD/MM/YYYY format.")

    dob_input = st.date_input(
        "Date of birth (DD/MM/YYYY)",
        value=st.session_state.get("date_of_birth", date(1980, 1, 1)),
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        format="DD/MM/YYYY",
    )

    encounter_input = st.date_input(
        "Date of encounter / current date (DD/MM/YYYY)",
        value=st.session_state.get("global_encounter_date", date.today()),
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        format="DD/MM/YYYY",
    )

    st.session_state["date_of_birth"] = dob_input
    st.session_state["global_encounter_date"] = encounter_input

    dob = dob_input
    encounter_date = encounter_input

    if is_future_date(encounter_date):
        st.error("Encounter / current date cannot be in the future. Please select today or an earlier date.")
        st.stop()

    if is_future_date(dob):
        st.error("Date of birth cannot be in the future.")
        st.stop()

    if dob > encounter_date:
        st.error("Date of birth cannot be after the encounter / current date.")
        st.stop()

    st.divider()

    st.subheader("Lymphocyte count")
    st.radio(
        "Lymphocyte count tested?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key="lymphocyte_tested",
    )

    if st.session_state["lymphocyte_tested"] == "Yes":
        c1, c2 = st.columns(2)
        with c1:
            st.date_input(
                "Date of test (DD/MM/YYYY)",
                value=st.session_state.get("lymphocyte_test_date", encounter_date),
                max_value=encounter_date,
                format="DD/MM/YYYY",
                key="lymphocyte_test_date",
            )
        with c2:
            st.number_input(
                "Lymphocyte count (×10⁹/L)",
                min_value=0.0,
                value=float(st.session_state.get("lymphocyte_count", 1.20)),
                step=0.1,
                format="%.2f",
                key="lymphocyte_count",
            )

    st.divider()

    st.subheader("CD19 count")
    st.radio(
        "CD19 tested?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key="cd19_tested",
    )

    if st.session_state["cd19_tested"] == "Yes":
        c1, c2 = st.columns(2)
        with c1:
            st.date_input(
                "CD19 test date (DD/MM/YYYY)",
                value=st.session_state.get("cd19_test_date", encounter_date),
                max_value=encounter_date,
                format="DD/MM/YYYY",
                key="cd19_test_date",
            )
        with c2:
            st.number_input(
                "CD19 value",
                min_value=0.0,
                value=float(st.session_state.get("cd19_value", 0.0)),
                step=0.1,
                format="%.2f",
                key="cd19_value",
            )

    st.divider()

    # --------------------------------------------------------
    # IV meds
    # --------------------------------------------------------
    for med_name, cfg in MEDS_IV.items():
        st.subheader(med_name)

        received = st.radio(
            f"Received {med_name}?",
            options=["No", "Yes"],
            index=0,
            horizontal=True,
            key=f"{med_name}_received",
        )

        if received == "No":
            st.caption("Not included (not received).")
            st.divider()
            continue

        st.number_input(
            "How many IV doses were given?",
            min_value=1,
            max_value=int(cfg["max_n_doses"]),
            value=int(st.session_state.get(f"{med_name}_n_doses", 1)),
            step=1,
            key=f"{med_name}_n_doses",
        )

        for i in range(int(st.session_state[f"{med_name}_n_doses"])):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    f"Dose #{i+1} ({cfg['units']})",
                    min_value=0,
                    value=int(st.session_state.get(f"{med_name}_dose_{i}", cfg["default_dose"])),
                    step=int(cfg["default_step"]),
                    format="%d",
                    key=f"{med_name}_dose_{i}",
                )
            with c2:
                st.date_input(
                    f"IV date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state.get(f"{med_name}_date_{i}", encounter_date),
                    max_value=encounter_date,
                    format="DD/MM/YYYY",
                    key=f"{med_name}_date_{i}",
                )

        st.divider()

    # --------------------------------------------------------
    # Oral Cyclophosphamide
    # --------------------------------------------------------
    st.subheader("Cyclophosphamide (Oral)")

    oral_received = st.radio(
        "Received Oral Cyclophosphamide?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key="oral_cyc_received",
    )

    if oral_received == "Yes":
        st.info(
            "If the medication has not been stopped yet, set the Stop date to the Encounter/Current date "
            "to estimate the current approximate score."
        )

        st.number_input(
            "How many oral cyclophosphamide courses were given?",
            min_value=1,
            max_value=int(ORAL_CYC["max_n_courses"]),
            value=int(st.session_state.get("oral_cyc_n_courses", 1)),
            step=1,
            key="oral_cyc_n_courses",
        )

        for i in range(int(st.session_state["oral_cyc_n_courses"])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state.get(f"oral_cyc_start_{i}", encounter_date),
                    max_value=encounter_date,
                    format="DD/MM/YYYY",
                    key=f"oral_cyc_start_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=bool(st.session_state.get(f"oral_cyc_not_stopped_{i}", False)),
                key=f"oral_cyc_not_stopped_{i}",
            )

            with c2:
                if st.session_state[f"oral_cyc_not_stopped_{i}"]:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=encounter_date,
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"oral_cyc_stop_disabled_{i}",
                    )
                else:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state.get(f"oral_cyc_stop_{i}", encounter_date),
                        max_value=encounter_date,
                        format="DD/MM/YYYY",
                        key=f"oral_cyc_stop_{i}",
                    )

            st.number_input(
                f"Daily dose #{i+1} ({ORAL_CYC['daily_dose_units']})",
                min_value=0,
                value=int(st.session_state.get(f"oral_cyc_daily_dose_{i}", 75)),
                step=25,
                format="%d",
                key=f"oral_cyc_daily_dose_{i}",
            )

            if i < int(st.session_state["oral_cyc_n_courses"]) - 1:
                st.divider()
    else:
        st.caption("Not included (not received).")

    st.divider()

    # --------------------------------------------------------
    # Azathioprine
    # --------------------------------------------------------
    render_decay_oral_medication_section(
        med_name="Azathioprine",
        cfg=AZATHIOPRINE,
        received_key="aza_received",
        n_courses_key="aza_n_courses",
        start_prefix="aza_start",
        stop_prefix="aza_stop",
        stop_disabled_prefix="aza_stop_disabled",
        not_stopped_prefix="aza_not_stopped",
        dose_prefix="aza_daily_dose",
    )

    st.divider()

    # --------------------------------------------------------
    # Mycophenolate mofetil
    # --------------------------------------------------------
    render_decay_oral_medication_section(
        med_name="Mycophenolate mofetil",
        cfg=MYCOPHENOLATE_MOFETIL,
        received_key="mmf_received",
        n_courses_key="mmf_n_courses",
        start_prefix="mmf_start",
        stop_prefix="mmf_stop",
        stop_disabled_prefix="mmf_stop_disabled",
        not_stopped_prefix="mmf_not_stopped",
        dose_prefix="mmf_daily_dose",
    )

    st.divider()

    # --------------------------------------------------------
    # Methotrexate
    # --------------------------------------------------------
    render_decay_oral_medication_section(
        med_name="Methotrexate",
        cfg=METHOTREXATE,
        received_key="meth_received",
        n_courses_key="meth_n_courses",
        start_prefix="meth_start",
        stop_prefix="meth_stop",
        stop_disabled_prefix="meth_stop_disabled",
        not_stopped_prefix="meth_not_stopped",
        dose_prefix="meth_dose",
    )

    st.divider()

    # --------------------------------------------------------
    # Tacrolimus
    # --------------------------------------------------------
    render_decay_oral_medication_section(
        med_name="Tacrolimus",
        cfg=TACROLIMUS,
        received_key="tac_received",
        n_courses_key="tac_n_courses",
        start_prefix="tac_start",
        stop_prefix="tac_stop",
        stop_disabled_prefix="tac_stop_disabled",
        not_stopped_prefix="tac_not_stopped",
        dose_prefix="tac_dose",
    )

    st.divider()

    # --------------------------------------------------------
    # Avacopan
    # --------------------------------------------------------
    st.subheader("Avacopan")
    st.radio(
        "Received Avacopan?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        key="avacopan_received",
    )
    if st.session_state["avacopan_received"] == "Yes":
        st.caption("Fixed dose 60 mg; fixed ITIS score 0.50.")
    else:
        st.caption("Not included (not received).")

    st.divider()

    # --------------------------------------------------------
    # Prednisolone
    # --------------------------------------------------------
    render_prednisolone_section()

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Back to introduction"):
            st.session_state.show_intro_page = True
            st.rerun()

    with c2:
        if st.button("Submit", type="primary"):
            st.session_state["date_of_birth"] = dob
            st.session_state["global_encounter_date"] = encounter_date

            result_payload = calculate_all_results()
            st.session_state.result_payload = result_payload
            st.session_state.show_result_page = True
            st.rerun()
