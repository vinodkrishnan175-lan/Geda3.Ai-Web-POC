
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import importlib
import traceback

st.set_page_config(page_title="Geda3.Ai Workshop TAT PoC", layout="wide")

ENGINE_ERROR = None
ENGINE_TRACEBACK = None
ENGINE_FILE = "unavailable"

try:
    tat_engine = importlib.import_module("tat_engine")
    list_sheets = tat_engine.list_sheets
    run_tat_pipeline = tat_engine.run_tat_pipeline
    read_table = getattr(tat_engine, "read_table", None)
    ENGINE_VERSION = getattr(tat_engine, "ENGINE_VERSION", "legacy-no-version")
    ENGINE_FILE = getattr(tat_engine, "__file__", "unknown")
except Exception as e:
    ENGINE_ERROR = e
    ENGINE_TRACEBACK = traceback.format_exc()
    ENGINE_VERSION = "engine-import-failed"
    list_sheets = None
    run_tat_pipeline = None
    read_table = None

CARD_CSS = """
<style>
.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 18px 20px;
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.08);
    min-height: 128px;
}
.metric-card .label {
    font-size: 0.95rem;
    opacity: 0.85;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .sub {
    margin-top: 8px;
    opacity: 0.8;
    font-size: 0.82rem;
}
.small-muted {
    color: #64748b;
    font-size: 0.88rem;
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.title("Geda3.Ai Workshop TAT PoC")
st.caption("Upload Gate Register + System/Tableau Extract, run the pipeline, review KPI cards, and download the workbook / CSV / ZIP outputs.")
st.caption(f"Engine build: {ENGINE_VERSION}")
st.caption(f"Engine module: {ENGINE_FILE}")

with st.expander("PoC note", expanded=True):
    st.write(
        "This version is designed for Streamlit Community Cloud as a client-facing proof of concept. "
        "Use masked or non-sensitive data for demos. Once the clients approve the logic and outputs, "
        "you can add authentication, secrets, audit logging, and secure deployment controls."
    )


if ENGINE_ERROR is not None:
    st.error("The app started, but tat_engine could not be imported.")
    st.exception(ENGINE_ERROR)
    with st.expander("Import traceback", expanded=True):
        st.code(ENGINE_TRACEBACK or "No traceback captured.")
    st.stop()


def _save_upload(uploaded, folder: str) -> str:
    path = Path(folder) / uploaded.name
    path.write_bytes(uploaded.getvalue())
    return str(path)


def _read_preview(uploaded, sheet_name=None):
    if uploaded.name.lower().endswith(("xlsx", "xlsm", "xls")):
        return pd.read_excel(uploaded, sheet_name=sheet_name)
    return pd.read_csv(uploaded)


def _metric_card(label: str, value, subtitle: str = ""):
    if value is None or value == "":
        display = "N/A"
    elif isinstance(value, float):
        display = f"{value:.2f}"
    else:
        display = str(value)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{display}</div>
            <div class="sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


left, right = st.columns(2)
with left:
    gate_file = st.file_uploader("1) Upload Gate Register", type=["xlsx", "xls", "xlsm", "csv"])
with right:
    sys_file = st.file_uploader("2) Upload System/Tableau Extract", type=["xlsx", "xls", "xlsm", "csv"])

gate_sheet = None
sys_sheet = None

if gate_file is not None:
    st.markdown("### Gate Register preview")
    try:
        if gate_file.name.lower().endswith(("xlsx", "xlsm", "xls")):
            with tempfile.TemporaryDirectory() as tmp:
                gate_tmp = _save_upload(gate_file, tmp)
                gate_sheets = list_sheets(gate_tmp)
            if gate_sheets:
                gate_sheet = st.selectbox("Gate Register sheet", gate_sheets, index=0)
        gate_preview = _read_preview(gate_file, gate_sheet)
        st.dataframe(gate_preview.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not preview Gate Register file: {e}")

if sys_file is not None:
    st.markdown("### System/Tableau preview")
    try:
        if sys_file.name.lower().endswith(("xlsx", "xlsm", "xls")):
            with tempfile.TemporaryDirectory() as tmp:
                sys_tmp = _save_upload(sys_file, tmp)
                sys_sheets = list_sheets(sys_tmp)
            if sys_sheets:
                sys_sheet = st.selectbox("System/Tableau sheet", sys_sheets, index=0)
        sys_preview = _read_preview(sys_file, sys_sheet)
        st.dataframe(sys_preview.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not preview System/Tableau file: {e}")

st.divider()

run_btn = st.button("Run Workshop TAT Analysis", type="primary", disabled=(gate_file is None or sys_file is None))

if run_btn:
    with st.spinner("Cleaning registrations, parsing dates, matching job cards, calculating TAT, validating business rules, and building outputs..."):
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = _save_upload(gate_file, tmp)
            sys_path = _save_upload(sys_file, tmp)
            out_dir = os.path.join(tmp, "outputs")
            os.makedirs(out_dir, exist_ok=True)

            try:
                result_paths = run_tat_pipeline(
                    gate_path=gate_path,
                    sys_path=sys_path,
                    out_dir=out_dir,
                    gate_sheet=gate_sheet,
                    sys_sheet=sys_sheet,
                )
            except Exception as e:
                st.error("Processing failed. Check the detected column mapping and the source date/time formats.")
                st.exception(e)
                st.stop()

            main_df = pd.read_csv(result_paths["main_csv"])
            summary_df = pd.read_csv(result_paths["summary_csv"])

            st.success("Analysis completed successfully.")

            with st.expander("Detected source columns", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Gate Register")
                    st.json(result_paths["detected_gate_columns"])
                with c2:
                    st.write("System/Tableau")
                    st.json(result_paths["detected_system_columns"])

            st.markdown("## Client KPI Summary")
            m = result_paths["summary_metrics"]

            a, b, c, d, e = st.columns(5)
            with a:
                _metric_card("Total gate vehicles\nNov 1–Dec 15", m["total_gate_vehicles_nov1_dec15"], "Gate Register rows inside operating window")
            with b:
                _metric_card("Valid gate entries", m["valid_gate_entries"], "Rows with calculable corrected TAT")
            with c:
                _metric_card("Matched job cards", m["matched_jcs"], "One-to-one matched system records")
            with d:
                _metric_card("Avg GIGO TAT", m["avg_gigo_tat_valid"], "Average corrected TAT for valid registrations")
            with e:
                _metric_card("Avg GIGO TAT delay", m["avg_gigo_delay_matched"], "Average delay for matched job cards")

            st.markdown("## Downloads")

            def _download_button(label: str, path: str, file_name: str | None = None, primary: bool = False):
                st.download_button(
                    label,
                    data=Path(path).read_bytes(),
                    file_name=file_name or Path(path).name,
                    type="primary" if primary else "secondary",
                )

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                _download_button("Download Excel workbook", result_paths["workbook"])
            with d2:
                _download_button("Download Main_Output CSV", result_paths["main_csv"])
            with d3:
                _download_button("Download Summary CSV", result_paths["summary_csv"])
            with d4:
                _download_button("Download ZIP bundle", result_paths["zip"], file_name="Workshop_TAT_Outputs.zip", primary=True)

            d5, d6 = st.columns(2)
            with d5:
                _download_button("Download Suggestions CSV", result_paths["suggestions_csv"])
            with d6:
                _download_button("Download Manual Review CSV", result_paths["manual_csv"])

            st.markdown("## Charts")
            c1, c2 = st.columns(2)
            with c1:
                st.image(result_paths["chart_match_distribution"], caption="Match distribution", use_container_width=True)
            with c2:
                st.image(result_paths["chart_tat_buckets"], caption="Corrected TAT buckets", use_container_width=True)

            st.image(result_paths["chart_waterfall"], caption="Pipeline overview", use_container_width=True)

            st.markdown("## Summary sheet preview")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("## Main output preview")
            st.dataframe(main_df.head(50), use_container_width=True)

            st.markdown(
                '<div class="small-muted">Tip: if a client file uses unusual headers, check the detected source columns expander first. '
                'The engine is already flexible, but header synonyms can be extended further in the next iteration.</div>',
                unsafe_allow_html=True,
            )
