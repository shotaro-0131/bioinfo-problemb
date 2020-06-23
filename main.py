from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import GetMACCSKeysFingerprint
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import Draw

mols = [m for m in Chem.SDMolSupplier('data/bmcmpd.sdf')]

maccs_fingerprint = [AllChem.GetMACCSKeysFingerprint(m) for m in mols]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)  for m in mols]


top_3 = [[0,0],[0,0],[0,0]]
top_score = [[0],[0],[0]]
print(top_3)
for i in range(len(mols)):
    for j in range(i,len(mols)):
        for k in range(3):
            differ = DataStructs.FingerprintSimilarity(ecfp[i],ecfp[j], metric=DataStructs.TanimotoSimilarity) - DataStructs.FingerprintSimilarity(maccs_fingerprint[i],maccs_fingerprint[j], metric=DataStructs.TanimotoSimilarity)

            if differ > top_score[k]:
                top_3[k][0] = i
                top_3[k][1] = j
                break
        
for i, t in enumerate(top_3):
    img=Draw.MolsToGridImage([mols[i] for i in t],molsPerRow=2,subImgSize=(200,200))
    
    img.save("mols_" + str(i) + ".pdf")    
    print("score is " + str(top_score[i]))