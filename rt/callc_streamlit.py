"""Streamlit-based web interface for CALLC."""

import base64
import logging
import os
import pathlib
from datetime import datetime
from importlib.metadata import version

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from streamlit_utils import hide_streamlit_menu, styled_download_button
from main import make_preds

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps

logger = logging.getLogger(__name__)


class CALLCStreamlitError(Exception):
    pass


class MissingCompoundCSV(CALLCStreamlitError):
    pass


class InvalidCompoundCSV(CALLCStreamlitError):
    pass


class InvalidCalibrationCompoundCSV(CALLCStreamlitError):
    pass


class MissingCalibrationCompoundCSV(CALLCStreamlitError):
    pass


class MissingCalibrationColumn(CALLCStreamlitError):
    pass


class StreamlitUI:
    """CALLC Streamlit UI."""

    def __init__(self):
        """CALLC Streamlit UI."""
        self.texts = WebpageTexts
        self.user_input = dict()

        st.set_page_config(
            page_title="CALLC web server",
            page_icon=":rocket:",
            layout="centered",
            initial_sidebar_state="expanded",
        )

        hide_streamlit_menu()

        self._main_page()
        self._sidebar()

    def _main_page(self):
        """Format main page."""
        st.title("CALLC")
        st.header("Input and configuration")
        st.subheader("Input files")
        self.user_input["input_csv"] = st.file_uploader(
            "Compound CSV", help=self.texts.Help.compound_csv
        )
        self.user_input["input_csv_calibration"] = st.file_uploader(
            "Calibration compound CSV",
            help=self.texts.Help.calibration_compound_csv,
        )
        """
        self.user_input["run_callc"] = st.checkbox(
            "Run CALLC"
        )
        self.user_input["run_kensert"] = st.checkbox(
            "Run Kensert et al."
        )
        self.user_input["run_bonini"] = st.checkbox(
            "Run Bonini et al."
        )
        self.user_input["run_yang"] = st.checkbox(
            "Run Yang et al."
        )
        self.user_input["slider_std"] = st.slider('What is the peak width for calculating overlap (%)', 0.0, 100.0, 0.1)
        """
        self.user_input["use_example"] = st.checkbox(
            "Use example data", help=self.texts.Help.example_data
        )

        with st.expander("Info about compound CSV formatting"):
            st.markdown(self.texts.Help.csv_formatting)

        if st.button("Predict retention times"):
            try:
                self._run_deeplc()
            except MissingCompoundCSV:
                st.error(self.texts.Errors.missing_compound_csv)
            except MissingCalibrationCompoundCSV:
                st.error(self.texts.Errors.missing_calibration_compound_csv)
            except MissingCalibrationColumn:
                st.error(self.texts.Errors.missing_calibration_column)
            except InvalidCompoundCSV:
                st.error(self.texts.Errors.invalid_compound_csv)
            except InvalidCalibrationCompoundCSV:
                st.error(self.texts.Errors.invalid_calibration_compound_csv)

    def _sidebar(self):
        """Format sidebar."""
        st.sidebar.image(
            "https://raw.githubusercontent.com/RobbinBouwmeester/CALLC/master/figs/logo.png",
            width=150,
        )
        st.sidebar.markdown(self.texts.Sidebar.badges)
        st.sidebar.header("About")
        st.sidebar.markdown(self.texts.Sidebar.about, unsafe_allow_html=True)

    def _run_deeplc(self):
        """Run CALLC given user input, and show results."""
        # Parse user config
        config = self._parse_user_config(self.user_input)

        logger.info(
            "Run requested // %s // compounds %i / use_library %r / calibrate %r",
            datetime.now(), len(config["input_df"])
        )

        # Run CALLC
        st.header("Running CALLC")
        status_placeholder = st.empty()
        status_placeholder.info(":hourglass_flowing_sand: Running CALLC...")
        
        if "\t" in config["input_df"][0].decode("utf-8"):
            struct_dict = dict([(v.decode("utf-8").split("\t")[0],v.decode("utf-8").split("\t")[1]) for v in config["input_df"]])
        else:
            struct_dict = dict([(v.decode("utf-8").split(",")[0],v.decode("utf-8").split(",")[1]) for v in config["input_df"]])

        try:
            preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs = make_preds(reference_infile=config["input_df_calibration"],pred_infile=config["input_df"],num_jobs=4,GUI_obj=None,ch_size=100000)
            #print(plot_setups)
        except Exception as e:
            status_placeholder.error(":x: CALLC ran into a problem")
            st.exception(e)
        else:
            status_placeholder.success(":heavy_check_mark: Finished!")

            # Add predictions to input DataFrame
            #result_df = pd.read_csv(self.user_input["input_csv"])

            result_df = preds_l3_test

            # Show head of result DataFrame
            st.header("Results")
            st.subheader("Selection of predicted retention times")
            st.dataframe(result_df.head(100))

            st.subheader("Coefficients of 'Layer 3'")

            coef_str = ""
            for m,coef in coefs:
                if coef < 0.025:
                    continue
                coef_str = coef_str+"%s -> %s \n\r" % (m.replace("+RtGAM",""),coef)
            st.write(coef_str)
            
            # Plot results
            self._plot_results(preds_l3_train, preds_l3_test, preds_l1_test.loc[:,plot_setups.index],struct_dict)

            # Download link
            st.subheader("Download predictions")
            filename = os.path.splitext(config["input_filename"])[0]
            self._df_download_href(result_df, filename + "_callc_predictions.csv")

    @staticmethod
    def get_example_input():
        """Return example DataFrame for input."""
        if os.path.isfile("example_data.csv"):
            example_df = pd.read_csv("example_data.csv")
        else:
            example_df = pd.DataFrame(
                [
                    ["AAGPSLSHTSGGTQSK", ""],
                    ["AAINQKLIETGER", "6|Acetyl"],
                    ["AANDAGYFNDEMAPIEVKTK", "12|Oxidation|18|Acetyl"],
                ],
                columns=["seq", "modifications"],
            )
        return example_df

    def _parse_user_config(self, user_input):
        """Validate and parse user input."""
        config = {
            "input_filename": None,
            "input_df": None,
            "input_df_calibration": None,
        }

        # Load example if use_example was selected
        if user_input["use_example"]:
            config["input_filename"] = "example.csv"
            config["input_df"] = self.get_example_input()
            config["input_df_calibration"] = config["input_df"]
            return config

        # Get compound dataframe
        if user_input["input_csv"]:
            config["input_filename"] = user_input["input_csv"].name
            try:
                config["input_df"] = user_input["input_csv"].readlines()
            except (ValueError, pd.errors.ParserError) as e:
                raise InvalidCompoundCSV(e)
        else:
            raise MissingCompoundCSV

        print(dir(user_input["input_csv_calibration"]))

        config["input_df_calibration"] = user_input["input_csv_calibration"].readlines()

        return config

    @staticmethod
    def _plot_results(result_df_train,result_df_test,lc_setups,struct_dict):
        """Plot results with Plotly Express."""
        
        st.subheader("Input retention times vs predictions")
        fig = px.scatter(
            result_df_train,
            x="predictions",
            y="tR",
            hover_data=[],
            trendline="ols",
            opacity=0.25,
            color_discrete_sequence=["#763737"],
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            xaxis_title_text="Input retention time",
            yaxis_title_text="Predicted retention time",
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted retention time distribution")
        fig = px.histogram(
            result_df_test,
            x="predictions",
            marginal="rug",
            opacity=0.8,
            nbins=80,
            histnorm="density",
            color_discrete_sequence=["#763737"],
        )
        fig.update_layout(
            xaxis_title_text="Predicted retention time",
            yaxis_title_text="Density",
            bargap=0.2,
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _df_download_href(df, filename="deeplc_predictions.csv"):
        """Get download href for pd.DataFrame CSV."""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        styled_download_button(
            "data:file/csv;base64," + b64,
            "Download results",
            download_filename=filename,
        )


class WebpageTexts:
    class Sidebar:
        badges = ""
        """
        [![GitHub release](https://img.shields.io/github/release-pre/compomics/deeplc.svg?style=flat-square)](https://github.com/compomics/deeplc/releases)
        [![GitHub](https://img.shields.io/github/license/compomics/deeplc.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
        [![Twitter](https://flat.badgen.net/twitter/follow/compomics?icon=twitter)](https://twitter.com/compomics)
        """

        about = f"""
            CALLC can accurately predict of liquid chromatographic retention times for small-molecule structures
            
            If you use CALLC for your research, please use the following citation:
            >**Generalized Calibration Across Liquid Chromatography Setups for Generic Prediction of Small-Molecule Retention Times**<br>
            >Robbin Bouwmeester, Lennart Martens, Sven Degroeve<br>
            >_Anal. Chem. (2020)_<br>
            >[doi:10.1021/acs.analchem.0c00233](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.0c00233)
            """

    class Help:
        compound_csv = """CSV with compounds for which to predict retention times. Click
            below on _Info about compound CSV formatting_ for more info.
            """
        calibration_compound_csv = """CSV with compounds with known retention times to be
            used for calibration. Click below on _Info about compound CSV formatting_ for
            more info.
            """
        example_data = "Use example data instead of uploaded CSV files."
        csv_formatting = """
            CALLC expects comma-separated values (CSV) with the following columns:

            - `IDENTIFIER`: identifier for compouynd
            - `preds`: smiles structure 
            - `time`: Retention time (only required for calibration CSV)

            For example:

            ```csv
            IDENTIFIER,preds,time
            LMGP01050041,CCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C,150
            LMGP01050045,CCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C,174
            LMGP06050006,CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC[C@@H](O)COP(=O)(O)O[C@H]1C(O)C(O)C(O)[C@@H](O)C1O,72
            LMGP01010573,CCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C,498
            LMGP01010976,CCCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCCC,582
            LMGP02011213,CCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCCC,510
            ```

            See
            [examples/datasets](https://github.com/RobbinBouwmeester/CALLC/tree/master/rt/datasets)
            for more examples.
            """
        calibration_source = """CALLC can calibrate its predictions based on set of
            known compound retention times. Calibration also ensures that the
            best-fitting CALLC model is used.
            """

    class Errors:
        missing_compound_csv = """
            Upload a compound CSV file or select the _Use example data_ checkbox.
            """
        missing_calibration_compound_csv = """
            Upload a calibration compound CSV file or select another _Calibration
            compounds_ option.
            """
        missing_calibration_column = """
            Upload a compound CSV file with a `tr` column or select another _Calibration
            compounds_ option.
            """
        invalid_compound_csv = """
            Uploaded compound CSV file could not be read. Click on _Info about compound
            CSV formatting_ for more info on the correct input format.
            """
        invalid_calibration_compound_csv = """
            Uploaded calibration compound CSV file could not be read. Click on _Info
            about compound CSV formatting_ for more info on the correct input format.
            """


if __name__ == "__main__":
    StreamlitUI()
