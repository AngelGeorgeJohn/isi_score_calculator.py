import streamlit as st
import numpy as np
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="ITIS Calculator", layout="centered")

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
    # A(dose)
    m_A, b_A = make_linear(cfg["A1"], cfg["A2"], cfg["dose1"], cfg["dose2"])
    A = m_A * dose + b_A

    # d(dose)
    m_d, b_d = make_linear(cfg["d1"], cfg["d2"], cfg["dose1"], cfg["dose2"])
    d = m_d * dose + b_d

    # n(dose)
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
    Course rule: include doses while (dose_date - course_start_date) <= window_days
    Returns list[(course_start_date, course_total_dose)]
    """
    if not entries:
        return []
    entries = sorted(entries, key=lambda x: x[0])

    courses = []
    course_start = entries[0][0]
    course_sum = entries[0][1]

    for dte, dse in entries[1:]:
        span = (dte - course_start).days
        if span <= window_days:
            course_sum += dse
        else:
            courses.append((course_start, course_sum))
            course_start = dte
            course_sum = dse

    courses.append((course_start, course_sum))
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
    "dose1": 75, "dose2": 25000,  # course_total model range
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
    global any_errors, overall_components, encounter_date

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

        n_courses = st.number_input(
            f"How many {med_name.lower()} courses were given?",
            min_value=1,
            max_value=int(cfg["max_n_courses"]),
            value=1,
            step=1,
            key=n_courses_key,
        )

        med_course_itises = []

        for i in range(int(n_courses)):
            st.markdown(f"**Course #{i+1}**")

            c1, c2 = st.columns(2)
            with c1:
                med_start = st.date_input(
                    f"Start date #{i+1} (DD/MM/YYYY)",
                    value=encounter_date,
                    max_value=encounter_date,
                    format="DD/MM/YYYY",
                    key=f"{start_prefix}_{i}",
                )

            med_not_stopped = st.checkbox(
                f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
                value=False,
                key=f"{not_stopped_prefix}_{i}",
            )

            with c2:
                if med_not_stopped:
                    med_stop = encounter_date
                    st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=med_stop,
                        disabled=True,
                        format="DD/MM/YYYY",
                        key=f"{stop_disabled_prefix}_{i}",
                    )
                else:
                    med_stop = st.date_input(
                        f"Stop date #{i+1} (DD/MM/YYYY)",
                        value=encounter_date,
                        max_value=encounter_date,
                        format="DD/MM/YYYY",
                        key=f"{stop_prefix}_{i}",
                    )

            med_daily_dose = st.number_input(
                f"Daily dose #{i+1} ({cfg['daily_dose_units']})",
                min_value=0,
                value=int(cfg["default_dose"]),
                step=int(cfg["default_step"]),
                format="%d",
                key=f"{dose_prefix}_{i}",
            )

            med_invalid = False
            if is_future_date(med_start) or is_after_encounter(med_start, encounter_date):
                st.error(
                    f"Course #{i+1}: Start date must be on or before the encounter/current date and cannot be a future date. "
                    f"This {med_name.lower()} course is excluded."
                )
                any_errors = True
                med_invalid = True

            if is_future_date(med_stop) or is_after_encounter(med_stop, encounter_date):
                st.error(
                    f"Course #{i+1}: Stop date must be on or before the encounter/current date and cannot be a future date. "
                    f"This {med_name.lower()} course is excluded."
                )
                any_errors = True
                med_invalid = True

            if med_stop < med_start:
                st.error(
                    f"Course #{i+1}: Stop date must be on or after start date. "
                    f"This {med_name.lower()} course is excluded."
                )
                any_errors = True
                med_invalid = True

            if int(med_daily_dose) < cfg["daily_min"] or int(med_daily_dose) > cfg["daily_max"]:
                st.error(
                    f"Course #{i+1}: Daily dose = {int(med_daily_dose)} is outside the allowed range "
                    f"({cfg['daily_min']}–{cfg['daily_max']}). "
                    f"This {med_name.lower()} course is excluded."
                )
                any_errors = True
                med_invalid = True

            if not med_invalid:
                # Because stop date max_value is encounter_date, encounter_date <= stop_date
                # is effectively equality for active treatment.
                if encounter_date <= med_stop:
                    med_itis = calculate_linear_score(
                        int(med_daily_dose),
                        cfg["dose1"],
                        cfg["dose2"],
                        cfg["min_score"],
                        cfg["max_score"],
                    )
                else:
                    interval_since_stop = (encounter_date - med_stop).days
                    if interval_since_stop < 0:
                        interval_since_stop = 0

                    med_itis = compute_itis(
                        interval_since_stop,
                        int(med_daily_dose),
                        cfg,
                    )

                med_course_itises.append(med_itis)

            if i < int(n_courses) - 1:
                st.divider()

        if med_course_itises:
            overall_components.append(combine_itis(med_course_itises))
        else:
            st.warning(f"No valid {med_name} courses were included.")
    else:
        st.caption("Not included (not received).")

# ============================================================
# UI
# ============================================================
st.title("Immunosuppressive Therapy Intensity Score (ITIS)")

st.subheader("Encounter / Current Date")
st.caption("Please enter/select dates in DD/MM/YYYY format.")

encounter_date = st.date_input(
    "Date of encounter / current date (DD/MM/YYYY)",
    value=date.today(),
    format="DD/MM/YYYY",
    key="global_encounter_date",
)

if is_future_date(encounter_date):
    st.error("Encounter / current date cannot be in the future. Please select today or an earlier date.")
    st.stop()

st.divider()

any_errors = False
overall_components = []

# ============================================================
# IV meds: per-dose validation ERROR; course total above upper => CLIP
# ============================================================
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

    n_doses = st.number_input(
        "How many IV doses were given?",
        min_value=1,
        max_value=int(cfg["max_n_doses"]),
        value=1,
        step=1,
        key=f"{med_name}_n_doses",
    )

    entries = []
    invalid_found = False

    for i in range(int(n_doses)):
        c1, c2 = st.columns(2)
        with c1:
            dose = st.number_input(
                f"Dose #{i+1} ({cfg['units']})",
                min_value=0,
                value=int(cfg["default_dose"]),
                step=int(cfg["default_step"]),
                format="%d",
                key=f"{med_name}_dose_{i}",
            )
        with c2:
            iv_date = st.date_input(
                f"IV date #{i+1} (DD/MM/YYYY)",
                value=encounter_date,
                max_value=encounter_date,
                format="DD/MM/YYYY",
                key=f"{med_name}_date_{i}",
            )

        if is_future_date(iv_date) or is_after_encounter(iv_date, encounter_date):
            st.error(
                f"{med_name} IV date #{i+1} must be on or before the encounter/current date and cannot be a future date."
            )
            any_errors = True
            invalid_found = True

        if not dose_is_valid(int(dose), cfg["dose1"], cfg["dose2"]):
            st.error(
                f"{med_name} Dose #{i+1} = {int(dose)} is outside the allowed range "
                f"({cfg['dose1']}–{cfg['dose2']}). {med_name} will be excluded."
            )
            any_errors = True
            invalid_found = True

        entries.append((iv_date, int(dose)))

    if invalid_found:
        st.warning(f"{med_name} was excluded due to invalid dose(s) and/or date(s).")
        st.divider()
        continue

    entries.sort(key=lambda x: x[0])
    courses = group_into_courses(entries, window_days=int(cfg["course_window_days"]))

    med_course_itises = []
    for course_start_date, course_sum_dose in courses:
        if is_future_date(course_start_date) or is_after_encounter(course_start_date, encounter_date):
            st.error(
                f"{med_name} course start date {date_display(course_start_date)} is invalid (after encounter/current date or future). "
                f"{med_name} is excluded."
            )
            any_errors = True
            med_course_itises = []
            break

        course_dose_used = clip_course_total(course_sum_dose, cfg["course_cap_dose"])

        if course_dose_used < float(cfg["course_min_dose"]):
            st.error(
                f"{med_name} course starting {date_display(course_start_date)} has cumulative dose "
                f"{int(course_dose_used)} (<{cfg['course_min_dose']}). This course is excluded."
            )
            any_errors = True
            continue

        days_since = (encounter_date - course_start_date).days
        if days_since < 0:
            st.error(
                f"Invalid dates for {med_name} course starting {date_display(course_start_date)}: "
                "encounter/current date must be on or after the IV date. This course is excluded."
            )
            any_errors = True
            continue

        med_course_itises.append(compute_itis(days_since, course_dose_used, cfg))

    if med_course_itises:
        overall_components.append(combine_itis(med_course_itises))
    else:
        st.warning(f"No valid {med_name} courses were included.")

    st.divider()

# ============================================================
# Oral Cyclophosphamide
# ============================================================
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

    n_oral_courses = st.number_input(
        "How many oral cyclophosphamide courses were given?",
        min_value=1,
        max_value=int(ORAL_CYC["max_n_courses"]),
        value=1,
        step=1,
        key="oral_cyc_n_courses",
    )

    oral_course_itises = []

    for i in range(int(n_oral_courses)):
        st.markdown(f"**Course #{i+1}**")

        c1, c2 = st.columns(2)
        with c1:
            oral_start = st.date_input(
                f"Start date #{i+1} (DD/MM/YYYY)",
                value=encounter_date,
                max_value=encounter_date,
                format="DD/MM/YYYY",
                key=f"oral_cyc_start_{i}",
            )

        not_stopped = st.checkbox(
            f"Course #{i+1} not stopped yet (use Encounter date as Stop date)",
            value=False,
            key=f"oral_cyc_not_stopped_{i}",
        )

        with c2:
            if not_stopped:
                oral_stop = encounter_date
                st.date_input(
                    f"Stop date #{i+1} (DD/MM/YYYY)",
                    value=oral_stop,
                    disabled=True,
                    format="DD/MM/YYYY",
                    key=f"oral_cyc_stop_disabled_{i}",
                )
            else:
                oral_stop = st.date_input(
                    f"Stop date #{i+1} (DD/MM/YYYY)",
                    value=encounter_date,
                    max_value=encounter_date,
                    format="DD/MM/YYYY",
                    key=f"oral_cyc_stop_{i}",
                )

        daily_dose = st.number_input(
            f"Daily dose #{i+1} ({ORAL_CYC['daily_dose_units']})",
            min_value=0,
            value=75,
            step=25,
            format="%d",
            key=f"oral_cyc_daily_dose_{i}",
        )

        oral_invalid = False
        if is_future_date(oral_start) or is_after_encounter(oral_start, encounter_date):
            st.error(
                f"Course #{i+1}: Start date must be on or before the encounter/current date and cannot be a future date. "
                "This oral cyclophosphamide course is excluded."
            )
            any_errors = True
            oral_invalid = True

        if is_future_date(oral_stop) or is_after_encounter(oral_stop, encounter_date):
            st.error(
                f"Course #{i+1}: Stop date must be on or before the encounter/current date and cannot be a future date. "
                "This oral cyclophosphamide course is excluded."
            )
            any_errors = True
            oral_invalid = True

        if oral_stop < oral_start:
            st.error(
                f"Course #{i+1}: Stop date must be on or after start date. "
                "This oral cyclophosphamide course is excluded."
            )
            any_errors = True
            oral_invalid = True

        if int(daily_dose) < ORAL_CYC["daily_min"] or int(daily_dose) > ORAL_CYC["daily_max"]:
            st.error(
                f"Course #{i+1}: Daily dose = {int(daily_dose)} is outside the allowed range "
                f"({ORAL_CYC['daily_min']}–{ORAL_CYC['daily_max']}). "
                "This oral cyclophosphamide course is excluded."
            )
            any_errors = True
            oral_invalid = True

        if not oral_invalid:
            effective_stop = min(oral_stop, encounter_date)
            days_on_drug = (effective_stop - oral_start).days
            course_total = float(days_on_drug) * float(int(daily_dose))

            if course_total < ORAL_CYC["course_min"]:
                st.error(
                    f"Course #{i+1}: Oral cyclophosphamide course total is {int(course_total)}, "
                    f"which is <{ORAL_CYC['course_min']}. This course is excluded."
                )
                any_errors = True
            else:
                course_total_used = clip_course_total(course_total, ORAL_CYC["course_max"])
                if course_total > ORAL_CYC["course_max"]:
                    st.info(
                        f"Course #{i+1}: Oral cyclophosphamide course_total (before cap) = {int(course_total)}. "
                        f"Using capped course_total = {int(course_total_used)}."
                    )

                interval_since_stop = (encounter_date - oral_stop).days
                if interval_since_stop < 0:
                    interval_since_stop = 0

                oral_course_itises.append(
                    compute_itis(interval_since_stop, course_total_used, ORAL_CYC)
                )

        if i < int(n_oral_courses) - 1:
            st.divider()

    if oral_course_itises:
        overall_components.append(combine_itis(oral_course_itises))
    else:
        st.warning("No valid Oral Cyclophosphamide courses were included.")
else:
    st.caption("Not included (not received).")

st.divider()

# ============================================================
# Azathioprine
# ============================================================
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

# ============================================================
# Mycophenolate mofetil
# ============================================================
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

# ============================================================
# Final result
# ============================================================
st.divider()

cumulative_itis = combine_itis(overall_components)
st.metric("Estimated Cumulative ITIS", f"{cumulative_itis:.4f}")

if any_errors:
    st.warning("One or more inputs were invalid. Some medications/courses may have been excluded.")
