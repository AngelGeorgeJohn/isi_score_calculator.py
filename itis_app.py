import streamlit as st
import numpy as np
from datetime import date

# -------------------------
# Core model helpers
# -------------------------
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

def warn_out_of_range(label, value, low, high):
    """Warn (but do not stop) if value is outside [low, high]."""
    if value < low or value > high:
        st.warning(f"{label} = {value:.0f} is outside the specified range ({low:.0f}–{high:.0f}).")

# -------------------------
# Medication configs (IV meds)
# -------------------------
MEDS_IV = {
    "Methylprednisolone": {
        "dose1": 250.0, "dose2": 3000.0,
        "A1": 0.40, "A2": 0.80,
        "d1": 15.0, "d2": 21.0,
        "vanish1": 21.0, "vanish2": 30.0,
        "units": "Dose units",
        "course_window_days": 30,
        "course_cap_dose": 3000.0,
        "course_min_dose": 250.0,
        "max_n_doses": 20,
        "default_step": 50.0,
        "default_dose": 250.0,
    },
    "Rituximab": {
        "dose1": 500.0, "dose2": 2000.0,
        "A1": 0.70, "A2": 0.85,
        "d1": 160.0, "d2": 200.0,
        "vanish1": 240.0, "vanish2": 300.0,
        "units": "Dose units",
        "course_window_days": 60,
        "course_cap_dose": 2000.0,
        "course_min_dose": 500.0,
        "max_n_doses": 20,
        "default_step": 100.0,
        "default_dose": 500.0,
    },
    "Cyclophosphamide (IV)": {
        "dose1": 150.0, "dose2": 8000.0,
        "A1": 0.60, "A2": 0.90,
        "d1": 40.0, "d2": 80.0,
        "vanish1": 58.0, "vanish2": 110.0,
        "units": "Dose units",
        "course_window_days": 180,
        "course_cap_dose": 8000.0,
        "course_min_dose": 150.0,
        "max_n_doses": 30,
        "default_step": 50.0,
        "default_dose": 150.0,
    },
}

# -------------------------
# Oral Cyclophosphamide config
# -------------------------
ORAL_CYC = {
    "dose1": 75.0, "dose2": 25000.0,
    "A1": 0.60, "A2": 0.90,
    "d1": 70.0, "d2": 90.0,
    "vanish1": 90.0, "vanish2": 115.0,
    "course_min": 75.0,
    "course_max": 25000.0,
    "daily_dose_units": "Daily dose units",
}

# -------------------------
# UI
# -------------------------
st.title("Immunosuppressive Therapy Intensity Score (ITIS)")

st.subheader("Encounter / Current Date")
encounter_date = st.date_input(
    "Date of encounter / current date (DD/MM/YYYY)",
    value=date.today(),
    key="global_encounter_date",
)

st.divider()

any_errors = False
overall_components = []

# =========================
# IV meds (course logic)
# =========================
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
    for i in range(int(n_doses)):
        c1, c2 = st.columns(2)
        with c1:
            dose = st.number_input(
                f"Dose #{i+1} ({cfg['units']})",
                min_value=0.0,
                value=float(cfg["default_dose"]),
                step=float(cfg["default_step"]),
                key=f"{med_name}_dose_{i}",
            )
        with c2:
            iv_date = st.date_input(
                f"IV date #{i+1} (DD/MM/YYYY)",
                value=date.today(),
                key=f"{med_name}_date_{i}",
            )

        # ✅ Warning if single entered dose outside specified model range
        warn_out_of_range(f"{med_name} Dose #{i+1}", dose, cfg["dose1"], cfg["dose2"])

        entries.append((iv_date, float(dose)))

    entries.sort(key=lambda x: x[0])

    if entries and (encounter_date - entries[0][0]).days < 0:
        st.error(f"Encounter/current date must be on or after the earliest {med_name} IV date.")
        any_errors = True
        st.divider()
        continue

    courses = group_into_courses(entries, window_days=int(cfg["course_window_days"]))

    med_course_itises = []
    for course_start_date, course_sum_dose in courses:
        # course total before cap (warn on this too)
        warn_out_of_range(f"{med_name} Course total (before cap)", course_sum_dose, cfg["dose1"], cfg["dose2"])

        # cap dose (your current behaviour)
        course_dose_capped = min(course_sum_dose, float(cfg["course_cap_dose"]))

        # warn on capped value as well (optional but useful)
        warn_out_of_range(f"{med_name} Course dose used", course_dose_capped, cfg["dose1"], cfg["dose2"])

        if course_dose_capped < float(cfg["course_min_dose"]):
            st.error(
                f"{med_name} course starting {course_start_date.strftime('%d/%m/%Y')} has cumulative dose "
                f"{course_dose_capped:.0f} (<{cfg['course_min_dose']:.0f}). This course is excluded."
            )
            any_errors = True
            continue

        if course_dose_capped > float(cfg["dose2"]):
            st.error(
                f"{med_name} course starting {course_start_date.strftime('%d/%m/%Y')} has cumulative dose "
                f"{course_dose_capped:.0f} which exceeds model upper bound ({cfg['dose2']:.0f}). "
                "This course is excluded."
            )
            any_errors = True
            continue

        days_since = (encounter_date - course_start_date).days
        if days_since < 0:
            st.error(
                f"Invalid dates for {med_name} course starting {course_start_date.strftime('%d/%m/%Y')}: "
                "encounter/current date must be on or after the IV date. This course is excluded."
            )
            any_errors = True
            continue

        med_course_itises.append(compute_itis(days_since, course_dose_capped, cfg))

    if med_course_itises:
        overall_components.append(combine_itis(med_course_itises))
    else:
        st.warning(f"No valid {med_name} courses were included.")

    st.divider()

# =========================
# Oral Cyclophosphamide
# =========================
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

    c1, c2 = st.columns(2)
    with c1:
        oral_start = st.date_input(
            "Start date (DD/MM/YYYY)",
            value=date.today(),
            key="oral_cyc_start",
        )

    not_stopped = st.checkbox(
        "Medication not stopped yet (use Encounter date as Stop date)",
        value=False,
        key="oral_cyc_not_stopped",
    )

    with c2:
        if not_stopped:
            oral_stop = encounter_date
            st.date_input(
                "Stop date (DD/MM/YYYY)",
                value=oral_stop,
                disabled=True,
                key="oral_cyc_stop_disabled",
            )
        else:
            oral_stop = st.date_input(
                "Stop date (DD/MM/YYYY)",
                value=date.today(),
                key="oral_cyc_stop",
            )

    daily_dose = st.number_input(
        f"Daily dose ({ORAL_CYC['daily_dose_units']})",
        min_value=0.0,
        value=75.0,
        step=25.0,
        key="oral_cyc_daily_dose",
    )

    if oral_stop < oral_start:
        st.error("Stop date must be on or after start date. Oral cyclophosphamide is excluded.")
        any_errors = True
    elif encounter_date < oral_start:
        st.error("Encounter/current date must be on or after the start date. Oral cyclophosphamide is excluded.")
        any_errors = True
    else:
        effective_stop = min(oral_stop, encounter_date)
        days_on_drug = (effective_stop - oral_start).days  # exclusive

        course_total = float(days_on_drug) * float(daily_dose)

        # ✅ Warning if course_total outside specified range BEFORE cap
        warn_out_of_range("Oral cyclophosphamide course_total (before cap)", course_total, ORAL_CYC["dose1"], ORAL_CYC["dose2"])

        if course_total < ORAL_CYC["course_min"]:
            st.error(
                f"Oral cyclophosphamide course total is {course_total:.0f}, which is <{ORAL_CYC['course_min']:.0f}. "
                "This course is excluded."
            )
            any_errors = True
        else:
            if course_total > ORAL_CYC["course_max"]:
                course_total = ORAL_CYC["course_max"]

            # ✅ Warning if capped value is still outside range (unlikely, but consistent)
            warn_out_of_range("Oral cyclophosphamide course_total used", course_total, ORAL_CYC["dose1"], ORAL_CYC["dose2"])

            interval_since_stop = (encounter_date - oral_stop).days
            if interval_since_stop < 0:
                interval_since_stop = 0

            overall_components.append(compute_itis(interval_since_stop, course_total, ORAL_CYC))
else:
    st.caption("Not included (not received).")

# Final cumulative ITIS
cumulative_itis = combine_itis(overall_components)
st.metric("Estimated Cumulative ITIS", f"{cumulative_itis:.4f}")

if any_errors:
    st.warning("One or more inputs were invalid. Some medications/courses may have been excluded.")


