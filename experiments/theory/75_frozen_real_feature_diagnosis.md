# Conditional diagnosis after the fixed real-packet comparison

Only admit this diagnostic if the completed eight-pass native stage fails to
dominate the selected strong control in development accuracy and NLL. A
running score is not the admission evidence. The existing fit, all controls,
source hashes and selected checkpoint stay unchanged.

Extract the32-dimensional query input to the native linear prediction head,
once per gesture, from both its selected encoder and its exact initial
encoder. Verify each original head reconstructs its saved probabilities and
that replay/readout fitting changes no encoder parameter or state. Both
encoders use identical common race draws and all984/192 saved causal clips.

Fit generic linear and RBF readouts on those features, using the same nine
fixed cells as the raw-packet calibration. Choose each encoder's readout by
3-fold fitting-only NLL, not the development result. Scaling is fitting-fold
only during decoder CV, then full-fitting only for its final fit. Preserve
every candidate CV result and selected readout artifact. The trained encoder
previously saw all fitting labels, so these conditional decoder folds are not
unbiased end-to-end validation; they only choose a frozen decoder. The192
subject-disjoint development examples remain exploratory after prior encoder
selection, and the official test stays untouched.

A substantially better readout of the same selected features would identify
usable information beyond its current decoder/optimization, without proving
that a simple readout change will learn end-to-end. A selected encoder better
than the identically probed initial encoder supports useful feature learning
under this probe protocol. Failure of both finite readout families does not
prove information absence. No difference establishes useful depth without a
matched shallow fit. Preserve these distinctions when interpreting results.

This is a frozen diagnostic control, not an architectural substitution or the
main research model. RBF support vectors add inference/storage costs; they do
not replace native temporal computation in the research thesis. Charge the
original encoder fit/workflow, complete feature replay, all decoder fits and
their calibration. Third-party solver arithmetic is unknown rather than zero.
Do not promote a diagnostic quality gain into practical advantage while
omitting these costs, latency, or the strongest raw-packet controls.
