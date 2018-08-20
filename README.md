# DepthConversion
## Seismic Depth Conversion using Quadratic Functions
* Given a oneway time depth pairs, a quadratic function is fitted  
resulting in 3 coefficients, _a0_, _a1_, and _a2_
* Given seismic horizon as an _x_, _y_, _z_ file, **ztqsegy.py**  
will convert the horizon from time to depth or depth to time.
* Quadratic coefficients can be just one set or supplied as three  
columns in one file or in three seperate files.
* Probabilities in the depth is computed from the computed  
_P10, P50_, and _P90_ of the corresponding coefficients. 

*  **ztqsegy.py** can also read in a segy in time or depth and convert  
that outputting another segy in the other domain.
