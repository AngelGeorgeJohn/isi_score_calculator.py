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

def dose_is_valid(dose, low, high):
    return (dose is not None) and (dose >= low) and (dose <= high)

def clip_course_total(course_total, upper):
    return min(float(course_total), float(upper))

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
        "units": "Dose units",
        "course_window_days": 30,
        "course_cap_dose": 3000,
        "course_min_dose": 250,
        "max_n_doses": 20,
        "default_step": 50,
        "default_dose": 250,
    },
    "Rituximab": {
        "dose1": 500, "dose2": 2000,
        "A1": 0.70, "A2": 0.85,
        "d1": 160.0, "d2": 200.0,
        "vanish1": 240.0, "vanish2": 300.0,
        "units": "Dose units",
        "course_window_days": 60,
        "course_cap_dose": 2000,
        "course_min_dose": 500,
        "max_n_doses": 20,
        "default_step": 100,
        "default_dose": 500,
    },
    "Cyclophosphamide (IV)": {
        "dose1": 150, "dose2": 8000,
        "A1": 0.60, "A2": 0.90,
        "d1": 40.0, "d2": 80.0,
        "vanish1": 58.0, "vanish2": 110.0,
        "units": "Dose units",
        "course_window_days": 180,
        "course_cap_dose": 8000,
        "course_min_dose": 150,
        "max_n_doses": 30,
        "default_step": 50,
        "default_dose": 150,
    },
}

ORAL_CYC = {
    "dose1": 75, "dose2": 25000,
    "A1": 0.60, "A2": 0.90,
    "d1": 70.0, "d2": 90.0,
    "vanish1": 90.0, "vanish2": 115.0,
    "course_min": 75,
    "course_max": 25000,
    "daily_dose_units": "Daily dose units",
    "daily_min": 0,
    "daily_max": 1000,
    "max_n_courses": 20,
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
    "daily_dose_units": "Daily dose units",
    "daily_min": 25,
    "daily_max": 250,
    "max_n_courses": 20,
    "default_dose": 25,
    "default_step": 25,
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
    "daily_dose_units": "Daily dose units",
    "daily_min": 125,
    "daily_max": 4000,
    "max_n_courses": 20,
    "default_dose": 125,
    "default_step": 125,
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
            value=1,
            step=1,
            key=n_courses_key,
        )

        for i in range(int(st.session_state[n_courses_key])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state["global_encounter_date"],
                    max_value=st.session_state["global_encounter_date"],
                    format="DD/MM/YYYY",
                    key=f"{start_prefix}_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=False,
                key=f"{not_stopped_prefix}_{i}",
            )

            with c2:
                if st.session_state[f"{not_stopped_prefix}_{i}"]:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"],
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"{stop_disabled_prefix}_{i}",
                    )
                else:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"],
                        max_value=st.session_state["global_encounter_date"],
                        format="DD/MM/YYYY",
                        key=f"{stop_prefix}_{i}",
                    )

            st.number_input(
                f"Daily dose #{i+1} ({cfg['daily_dose_units']})",
                min_value=0,
                value=int(cfg["default_dose"]),
                step=int(cfg["default_step"]),
                format="%d",
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
            value=1,
            step=1,
            key="prd_n_courses",
        )

        for i in range(int(st.session_state["prd_n_courses"])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=st.session_state["global_encounter_date"],
                    max_value=st.session_state["global_encounter_date"],
                    format="DD/MM/YYYY",
                    key=f"prd_start_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=False,
                key=f"prd_not_stopped_{i}",
            )

            with c2:
                if st.session_state[f"prd_not_stopped_{i}"]:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"],
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"prd_stop_disabled_{i}",
                    )
                else:
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=st.session_state["global_encounter_date"],
                        max_value=st.session_state["global_encounter_date"],
                        format="DD/MM/YYYY",
                        key=f"prd_stop_{i}",
                    )

            st.selectbox(
                f"Dose category #{i+1}",
                options=PREDNISOLONE["dose_categories"],
                index=PREDNISOLONE["dose_categories"].index(PREDNISOLONE["default_category"]),
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
            dose = int(st.session_state.get(f"{med_name}_dose_{i}", cfg["default_dose"]))
            iv_date = st.session_state.get(f"{med_name}_date_{i}", encounter_date)

            if is_future_date(iv_date) or is_after_encounter(iv_date, encounter_date):
                any_errors = True
                invalid_found = True

            if not dose_is_valid(dose, cfg["dose1"], cfg["dose2"]):
                any_errors = True
                invalid_found = True

            entries.append((iv_date, dose))
            med_entered_doses.append(f"{date_display(iv_date)}: {dose}")

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

            # Use LAST dose date as IV reference date
            days_since = (encounter_date - course_last_date).days

            # Replace negative interval with zero
            if days_since < 0:
                days_since = 0

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
            daily_dose = int(st.session_state.get(f"oral_cyc_daily_dose_{i}", 75))

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
            if daily_dose < ORAL_CYC["daily_min"] or daily_dose > ORAL_CYC["daily_max"]:
                any_errors = True
                oral_invalid = True

            if not oral_invalid:
                effective_stop = min(oral_stop, encounter_date)
                days_on_drug = (effective_stop - oral_start).days
                course_total = float(days_on_drug) * float(daily_dose)

                if course_total < ORAL_CYC["course_min"]:
                    any_errors = True
                    oral_summary.append(
                        f"course #{i+1}: excluded (entered {daily_dose}/day, {date_display(oral_start)} to {date_display(oral_stop)})"
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
                        f"course #{i+1}: {date_display(oral_start)} to {date_display(oral_stop)}, dose {daily_dose}/day"
                    )
            else:
                oral_summary.append(f"course #{i+1}: excluded due to invalid input(s).")

        if oral_course_itises:
            overall_components.append(combine_itis(oral_course_itises))
            summary_lines.append("- Cyclophosphamide (Oral): " + "; ".join(oral_summary))
        else:
            summary_lines.append("- Cyclophosphamide (Oral): no valid course included.")

    # --------------------------------------------------------
    # Azathioprine / MMF
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
        nonlocal any_errors, overall_components, summary_lines, encounter_date

        if st.session_state.get(received_key, "No") != "Yes":
            return

        n_courses = int(st.session_state.get(n_courses_key, 1))
        med_course_itises = []
        med_summary = []

        for i in range(n_courses):
            med_start = st.session_state.get(f"{start_prefix}_{i}", encounter_date)
            med_not_stopped = st.session_state.get(f"{not_stopped_prefix}_{i}", False)
            med_stop = encounter_date if med_not_stopped else st.session_state.get(f"{stop_prefix}_{i}", encounter_date)
            med_daily_dose = int(st.session_state.get(f"{dose_prefix}_{i}", cfg["default_dose"]))

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
            if med_daily_dose < cfg["daily_min"] or med_daily_dose > cfg["daily_max"]:
                any_errors = True
                med_invalid = True

            if not med_invalid:
                if encounter_date <= med_stop:
                    med_itis = calculate_linear_score(
                        med_daily_dose,
                        cfg["dose1"],
                        cfg["dose2"],
                        cfg["min_score"],
                        cfg["max_score"],
                    )
                else:
                    interval_since_stop = (encounter_date - med_stop).days
                    if interval_since_stop < 0:
                        interval_since_stop = 0

                    med_itis = compute_itis(interval_since_stop, med_daily_dose, cfg)

                med_course_itises.append(med_itis)
                med_summary.append(
                    f"course #{i+1}: {date_display(med_start)} to {date_display(med_stop)}, dose {med_daily_dose}/day"
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
        "cumulative_itis": cumulative_itis,
        "any_errors": any_errors,
        "summary_lines": summary_lines,
    }

# ============================================================
# Result page
# ============================================================
if st.session_state.show_result_page and st.session_state.result_payload is not None:
    result = st.session_state.result_payload

    st.title("Estimated Cumulative ITIS Result")
    st.caption(f"Encounter / Current Date: {date_display(result['encounter_date'])}")

    st.metric("Estimated Cumulative ITIS", f"≈ {result['cumulative_itis']:.2f}")

    st.subheader("Summary of Entered Medications")
    if result["summary_lines"]:
        for line in result["summary_lines"]:
            st.write(line)
    else:
        st.write("No medications were entered.")

    if result["any_errors"]:
        st.warning("One or more inputs were invalid. Some medications/courses may have been excluded.")

    if st.button("Back to entry form"):
        st.session_state.show_result_page = False
        st.rerun()

# ============================================================
# Entry page
# ============================================================
else:
    st.title("Immunosuppressive Therapy Intensity Score (ITIS)")
    st.subheader("Encounter / Current Date")
    st.caption("Please enter/select dates in DD/MM/YYYY format.")

    st.date_input(
        "Date of encounter / current date (DD/MM/YYYY)",
        value=date.today(),
        format="DD/MM/YYYY",
        key="global_encounter_date",
    )

    encounter_date = st.session_state["global_encounter_date"]

    if is_future_date(encounter_date):
        st.error("Encounter / current date cannot be in the future. Please select today or an earlier date.")
        st.stop()

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
            value=1,
            step=1,
            key=f"{med_name}_n_doses",
        )

        for i in range(int(st.session_state[f"{med_name}_n_doses"])):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    f"Dose #{i+1} ({cfg['units']})",
                    min_value=0,
                    value=int(cfg["default_dose"]),
                    step=int(cfg["default_step"]),
                    format="%d",
                    key=f"{med_name}_dose_{i}",
                )
            with c2:
                st.date_input(
                    f"IV date #{i+1} (DD/MM/YYYY)",
                    value=encounter_date,
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
            value=1,
            step=1,
            key="oral_cyc_n_courses",
        )

        for i in range(int(st.session_state["oral_cyc_n_courses"])):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=encounter_date,
                    max_value=encounter_date,
                    format="DD/MM/YYYY",
                    key=f"oral_cyc_start_{i}",
                )

            st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=False,
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
                        value=encounter_date,
                        max_value=encounter_date,
                        format="DD/MM/YYYY",
                        key=f"oral_cyc_stop_{i}",
                    )

            st.number_input(
                f"Daily dose #{i+1} ({ORAL_CYC['daily_dose_units']})",
                min_value=0,
                value=75,
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
    # Prednisolone
    # --------------------------------------------------------
    render_prednisolone_section()

    st.divider()

    if st.button("Submit", type="primary"):
        result_payload = calculate_all_results()
        st.session_state.result_payload = result_payload
        st.session_state.show_result_page = True
        st.rerun()
