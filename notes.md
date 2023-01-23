# Segmentation 

- Projection verticale
- A l'intérieur d'un bloc déterminé par la projection verticale : 
    - on trouve les caractères comme les étant les composantes connexes ; ou bien en faisant une projection horizontale ;
    - les caractères ayant une plus "petite" surface sont regroupés avec leur plus proche voisin comme étant des caractères "doubles" --> permet de reconnaître les $=$, $x^2$, $+\infty$...

❓ Comment reconnaître les caractères groupés qui ne sont ni connexes ni doubles, comme $\lim$ ? Et distinguer ceux qui sont connexes sans être un seul caractère, comme $\frac{y}{x}$ ?

# Recognition

- Template matching
- Image descriptor
- Random Forest

# Etape 1

Dataset : CHROHME (converti en offline)   
Recognition of isolated math symbols => once a symbol is isolated, we put it into Random Forest classifier

VIC is used for : 
- Preprocessing
- Detect individual characters in a line

Datasets d'équations handwritten : 
https://ai.100tal.com/dataset --> il faut un numéro de tel chinois !!! J'arrive pas à me connecter
CROHME = écrit avec une tablette, pas un stylo sur une feuille

Datasets de symboles handwritten isolés : 
https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols

Formula is in black and the background is transparent ; ATTENTION, latex pdf NOT handwritten :
https://zenodo.org/record/56198#.V2p0KTXT6eA

Code Im to Latex (based on above dataset) :
https://github.com/harvardnlp/im2markup

Dans ce papier : https://shchae7.github.io/pdfs/handwritten-math-formula-recogntion.pdf Ils créent leur propre dataset en ajoutant un fond derrière les images