"""Streamlit-based web interface for CALLC."""

import base64
import logging
import os
import pathlib
from datetime import datetime
#from importlib.metadata import version

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from matplotlib import pyplot as plt

from streamlit_utils import hide_streamlit_menu, styled_download_button
from main import make_preds

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
import pickle
import seaborn as sns

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

def get_optimal_sep(df_setup,expected_grad_length=0.0):
    n_compounds = len(df_setup.index)
    optimal_dist = 1.0/n_compounds
    tot_error_dict = {}
    for col in df_setup.columns:
        min_tr = min(df_setup[col])
        max_tr = max(df_setup[col])-min_tr
        
        tr_list = sorted(list((df_setup[col]-min_tr)/max_tr))
        error_list = []
        print(tr_list)
        tot_error_dict[col] = 0.0

        for idx,list_val in enumerate(tr_list):
            if idx < n_compounds-1:
                tot_error_dict[col] += abs(optimal_dist-abs(list_val-tr_list[idx+1]))
                error_list.append(abs(optimal_dist-abs(list_val-tr_list[idx+1])))
        print(error_list)
        print(sum(error_list))
        print(len(df_setup.columns))
        print("+++++++++++++++++")
        tot_error_dict[col] = tot_error_dict[col]/len(tr_list)

        #tot_error_dict[col] += abs(max_tr-expected_grad_length)/expected_grad_length

    print(tot_error_dict)
    return tot_error_dict

def get_overlapping_compounds(result_df_test,struct_dict,dist=0.05):
    temp_result_df_test = result_df_test.sort_values(by="predictions")
    max_pred = max(temp_result_df_test["predictions"])

    predicted_overlap = []
    prev_analyzed = set([])

    for i1 in range(len(temp_result_df_test)):
        for i2 in range(len(temp_result_df_test)):
            if i1 == i2:
                continue
            if i1 > i2:
                continue
            if abs(temp_result_df_test.iloc[i1]["predictions"]-temp_result_df_test.iloc[i2]["predictions"])/max_pred < dist:
                if struct_dict[temp_result_df_test.iloc[i1]["identifiers"]] == struct_dict[temp_result_df_test.iloc[i2]["identifiers"]]:
                    continue
                if temp_result_df_test.iloc[i1]["identifiers"]+"|"+temp_result_df_test.iloc[i2]["identifiers"] in prev_analyzed:
                    continue

                predicted_overlap.append([struct_dict[temp_result_df_test.iloc[i1]["identifiers"]],
                                          temp_result_df_test.iloc[i1]["predictions"],
                                          struct_dict[temp_result_df_test.iloc[i2]["identifiers"]],    
                                          temp_result_df_test.iloc[i2]["predictions"],
                                          temp_result_df_test.iloc[i1]["identifiers"],
                                          temp_result_df_test.iloc[i2]["identifiers"]
                                        ])
                prev_analyzed.add(temp_result_df_test.iloc[i1]["identifiers"]+"|"+temp_result_df_test.iloc[i2]["identifiers"])
                prev_analyzed.add(temp_result_df_test.iloc[i2]["identifiers"]+"|"+temp_result_df_test.iloc[i1]["identifiers"])

    return predicted_overlap

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
        st.subheader("Run parameters")
        self.user_input["run_callc"] = st.checkbox(
            "Run CALLC models"
        )
        self.user_input["run_kensert"] = st.checkbox(
            "Run Kensert et al.  models"
        )
        self.user_input["run_bonini"] = st.checkbox(
            "Run Bonini et al. models"
        )
        self.user_input["run_yang"] = st.checkbox(
            "Run Yang et al. models"
        )
        
        self.user_input["slider_std"] = st.slider('What is the peak width for calculating overlap (%)', 0.0, 25.0, 3.0)

        self.user_input["include_future"] = st.checkbox(
            "Include models for future use", help=self.texts.Help.include_future
        )

        st.markdown("""---""")

        self.user_input["use_example"] = st.checkbox(
            "Use example data", help=self.texts.Help.example_data
        )

        with st.expander("Info about compound CSV formatting"):
            st.markdown(self.texts.Help.csv_formatting)

        st.markdown("""---""")

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
            struct_dict = dict([(v.decode("utf-8").split("\t")[5],v.decode("utf-8").split("\t")[-2]) for v in config["input_df"]])
        else:
            struct_dict = dict([(v.decode("utf-8").split(",")[0],v.decode("utf-8").split(",")[1]) for v in config["input_df"]])


        try:
            

            #f = open('plotting_pickle_v2.p', 'wb')
            #pickle.dump([preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs, test_df], f)
            #f.close()
            if config["input_filename"].endswith("degrad_6_sub2.tsv"):
                f = open("plotting_pickle_v2.p", "rb")
                preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs, test_df = pickle.load(f)
                f.close()
            elif config["input_filename"].endswith("degrad_6_sub.tsv"):
                f = open("plotting_pickle.p", "rb")
                preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs, test_df = pickle.load(f)
                f.close()
            else:
                preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs, test_df = make_preds(reference_infile=config["input_df_calibration"],pred_infile=config["input_df"],num_jobs=4,GUI_obj=None,ch_size=100000)
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
            with st.expander("Selection of predicted retention times"):
                st.dataframe(result_df.head(100))

            with st.expander("Coefficients of 'Layer 3'"):
                coef_str = ""
                for m,coef in coefs:
                    if coef < 0.025:
                        continue
                    coef_str = coef_str+"%s → %s \n\r" % (m.replace("+RtGAM",""),round(coef,4))
                st.write(coef_str)
            
            # Plot results
            self._plot_results(preds_l3_train, preds_l3_test, preds_l1_test.loc[:,plot_setups.index],struct_dict,test_df)

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
                    
                    ["LMGP01050041","CCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",150],
                    ["LMGP01050045","CCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",174],
                    ["LMGP06050006","CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC[C@@H](O)COP(=O)(O)O[C@H]1C(O)C(O)C(O)[C@@H](O)C1O",72],
                    ["LMGP01010573","CCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",498],
                    ["LMGP01010976","CCCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCCC",582],
                    ["LMGP02011213","CCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCCC",510]
                ],
                columns=["IDENTIFIER","preds","time"],
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
    def _plot_results(result_df_train,result_df_test,lc_setups,struct_dict,test_df):
        """Plot results with Plotly Express."""
        
        with st.expander("Predictions for the calibration set"):
            plt.scatter(result_df_train["predictions"],result_df_train["tR"])
            plt.plot([min([min(result_df_train["tR"]),min(result_df_train["predictions"])]),
                      max([max(result_df_train["tR"]),max(result_df_train["predictions"])])],
                     [min([min(result_df_train["tR"]),min(result_df_train["predictions"])]),
                      max([max(result_df_train["tR"]),max(result_df_train["predictions"])])],
                      linestyle="--",
                      color="grey",
                      zorder=0,
                      linewidth=1)
            plt.xlabel("Predicted retention time")
            plt.ylabel("Observed retention time")
            fig = plt.gcf()
            st.pyplot(fig)
            
            plt.close()

        """
        with st.expander("Umap with previous training data"):
            embedding = pickle.load(open("umap/embedding.p","rb"))
            sel_feats = pickle.load(open("umap/sel_feats.p","rb"))
            umap_model = pickle.load(open("umap/umap_model.p","rb"))
            embedding_test = umap_model.transform(test_df[sel_feats])
            plt.scatter(embedding[:,0],embedding[:,1],s=1,alpha=0.2,label="Seen compounds")
            plt.scatter(embedding_test[:,0],embedding_test[:,1],label="Prediction compounds")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            leg = plt.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
                print(dir(lh))
                lh.set_sizes([20])
            fig2 = plt.gcf()
            st.pyplot(fig2)
            plt.close()
        """

        with st.expander("Predictions for the input data"):
            plt.hist(result_df_test["predictions"], bins=80)
            plt.xlabel("Predicted retention time")
            plt.ylabel("Frequency (#)")
            fig2 = plt.gcf()
            st.pyplot(fig2)
            plt.close()

        with st.expander("Simulated chromatogram"):
            max_tr = max(result_df_test["predictions"])
            perc_diff = max_tr*0.05
            density_vals = []
            
            for v in result_df_test["predictions"]:
                for i in range(50):
                    density_vals.append(np.random.normal(v, perc_diff))
            fig3 = plt.figure(figsize=(10, 4))
            sns.distplot(density_vals, 
                        hist = False, kde = True,
                        kde_kws = {'linewidth': 3})
            plt.xlabel("Retention time")
            st.pyplot(fig3)
            plt.close()

        with st.expander("Predicted overlapping elution profiles"):
            max_pred = max(result_df_test["predictions"])
            overlap_list = get_overlapping_compounds(result_df_test,struct_dict)
            for entry in overlap_list:
                
                img=Draw.MolsToGridImage([Chem.MolFromSmiles(entry[0]),Chem.MolFromSmiles(entry[2])],
                        molsPerRow=2,subImgSize=(300,300),legends=[entry[4],entry[5]])
                
                st.image(img, caption='overlap')

                density_vals_1 = []
                density_vals_2 = []
                #for v in :
                for i in range(250):
                    density_vals_1.append(np.random.normal(entry[1], max_pred*0.01))
                    density_vals_2.append(np.random.normal(entry[3], max_pred*0.01))
                fig3 = plt.figure(figsize=(8, 1))
                sns.kdeplot(density_vals_1,fill=True)
                sns.kdeplot(density_vals_2,fill=True)
                plt.xlabel("Retention time")
                st.pyplot(fig3)
                plt.close()
                st.markdown("""---""")


        with st.expander("Suggestions alternative LC setup"):
            suggested_lc = get_optimal_sep(lc_setups,expected_grad_length=max(result_df_test["predictions"]))
            suggested_lc = sorted(suggested_lc.items(), key=lambda x: x[1], reverse=False)

            for l,name_score in enumerate(suggested_lc):
                name,score = name_score
                new_name = name.replace("_xgb","")
                fig4 = plt.figure(figsize=(10, 4))

                st.markdown(f"**Name: {new_name}**")
                
                sns.distplot(lc_setups[name], 
                            hist = True, kde = True,
                            kde_kws = {'linewidth': 3})
                plt.title(new_name)
                st.pyplot(fig4)
                plt.close()
                st.markdown("""---""")
                if l > 10:
                    break

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
        include_future = "The models trained on the calibration compounds will be saved and used for future predictions"
        csv_formatting = """
            CALLC expects comma-separated values (CSV) with the following columns:

            - `IDENTIFIER`: identifier for compouynd
            - `inchi`: smiles or inchi structure 
            - `time`: Retention time (only required for calibration CSV)

            For example:

            ```csv
            IDENTIFIER,inchi,time
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
