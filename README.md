# Thor
Titers from High-Output Reads is a python library for calculating replicative fitness and titers
from sequencing reads. The main model can be imported from the models.py module via

  from Thor import BB_model

The main input to this model is a table who should contain the columns: SERUM, REPEAT, DILUTION, CT
and any variants for which counts are reported. SERUM, REPEAT, DILUTION give the relevant information
for the sample for each row where as CT should be the value of the CT measurement (or analogous such as
-log(PFU)) of the supernatat after the assay. SERUM should contain INPUT and NO SERUM for which dilution
values should be empty strings. Different samples can have different REPEAT names. For instance
if INPUT has a single repeat, its REPEAT value can be an empty string where as if NO SERUM has three repeats
its repeat values can be A,B. Other relevant input to the model are explained in the doc string. More details
about how the model was formulated and how it works is given in the paper [PAPER NAME](link). For an example
table see tests/tests_input folder.

Functions inside samplers.py module can be used to carry out posterior predictive sampling for various
observables of interest such as pairwise differences between titers or neutralization curves.

Functions inside simulator.py module can be used to simulate datasets for testing.

Functions inside nonparametric.py are used to obtain non-parametric estimates for replicative fitness
and titers (see the paper above for details) which are then used to vaguely inform priors by using
population averages and standard deviations of these parameters.

## Logging

There isn't an intensive logging activity in Thor, critical errors are dealt
with exceptions whereas user ammendable minor problems are handled with warnings.
Nevertheless there is some logging done for INFO and for ERROR type events that are possibly
fixable without user intervention. You can use set_logger_props to set logger level and add
streams. You can also edit logging.config (there is only a single logger called
Thor.models).

## Internal Errors

If you get an InternalError exception, it means that Thor has somehow
messed something during preprocessing stage (could still be related to
unexpected inputs not handled gracefully). If that is the case please
raise an issue with relevant details and a version of your data that can
still generate the error (the error will almost always be related to
SERUM,REPEAT,DILUTION values so feel free to set all the rest of the data
to nan).
