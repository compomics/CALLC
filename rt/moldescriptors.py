from rdkit import Chem
from getf import getf

def read_lib_identifiers(infile):
    infile = open(infile)
    identifiers = []
    for line in infile:
        if line.startswith("IDENTIFIER"): continue
        line = line.strip()
        identifiers.append(line.split(",")[0])
    return(set(identifiers))

def get_feats_lib(infile,extract_list):
    infile = open(infile)
    ret_dict = {}
    unique_smiles = set([j for i,j in extract_list])
    smiles_to_name = dict([(j,i) for i,j in extract_list])
    for line in infile:
        if line.startswith("IDENTIFIER"): header = line.strip().split(",")
        line = line.rstrip()
        identifier = line.split(",")[0]
        if identifier in unique_smiles:
            split_line = line.split(",")[1:]
            ret_dict[smiles_to_name[identifier]] = dict(zip(header[1:],split_line))
    return(ret_dict)

def get_features(infile_name="data/LMSDFDownload28Jun15FinalAll.sdf",outfile_name="lmfeatures.csv",library_file="feats_lib.csv",
                 id_index=0,mol_index=1,time_index=None,gui_object=None,chunk_size=10000):

    outfile = open(outfile_name,"w")

    features = []
    library_analyze_smiles = []
    lipid_dict = {}
    time_dict = {}
    counter = 0
    write_header = True

    if gui_object: tot_to_analyze = float(len(open(infile_name).readlines()))

    if len(library_file.strip()) > 0: lib_identifiers = read_lib_identifiers(library_file)
    else: lib_identifiers = set()

    mols = open(infile_name)

    
    for mol in mols:
        counter += 1

        if gui_object:
            perc = round((float(counter)/tot_to_analyze)*100.0,1)
            if perc.is_integer():
                gui_object.update_progress2(perc)

        if (counter % chunk_size) == 0 and counter != 0:
            if len(library_analyze_smiles) > 0: 
                lib_dict = get_feats_lib(library_file,library_analyze_smiles)
                lipid_dict.update(lib_dict)

            if len(features) == 0:
                features = list(lipid_dict[list(lipid_dict.keys())[0]].keys())

            if write_header:
                if time_index: outfile.write("IDENTIFIER,time,%s\n" % (",".join(features)))
                else: outfile.write("IDENTIFIER,%s\n" % (",".join(features)))
                write_header = False

            for identifier in lipid_dict.keys():
                if time_index:
                    outfile.write("%s," % (identifier))
                    outfile.write("%s" % (time_dict[identifier]))
                else: outfile.write("%s" % (identifier))
                for f in features:
                    outfile.write(",%s" % (lipid_dict[identifier][f]))
                outfile.write("\n")
            #outfile.close()
            library_analyze_smiles = []
            lipid_dict = {}
            time_dict = {}

        if "\t" in mol: mol = mol.strip().split("\t")
        else: mol = mol.strip().split(",")

        identifier = mol[id_index]
        mol_str = mol[mol_index]
        if mol_str == "SMILES": continue
        if time_index: 
            rt = mol[time_index]
            time_dict[identifier] = rt

        m = Chem.MolFromSmiles(mol_str)

        if m == None: 
            #TODO write error msg
            continue
        if mol_str not in lib_identifiers:
            try: fdict = getf(m)
            except: continue
            lipid_dict[identifier] = fdict["rdkit"]
            if len(features) == 0: features = list(set(lipid_dict[identifier].keys()))
        else:
            library_analyze_smiles.append([identifier,mol_str])
    
    if len(library_analyze_smiles) > 0: 
        lib_dict = get_feats_lib(library_file,library_analyze_smiles)
        lipid_dict.update(lib_dict)

    if len(features) == 0:
        features = list(lipid_dict[list(lipid_dict.keys())[0]].keys())

    if write_header:
        if time_index: outfile.write("IDENTIFIER,time,%s\n" % (",".join(features)))
        else: outfile.write("IDENTIFIER,%s\n" % (",".join(features)))
        write_header = False

    for identifier in lipid_dict.keys():
        
        if time_index:
            outfile.write("%s," % (identifier))
            outfile.write("%s" % (time_dict[identifier]))

        else: outfile.write("%s" % (identifier))
        for f in features:
            outfile.write(",%s" % (lipid_dict[identifier][f]))
        outfile.write("\n")
    outfile.close()


if __name__ == "__main__":
    get_features()