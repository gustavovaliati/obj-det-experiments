# Tasks
* Test with 16x16
* Test with 640x480

# Net Design

How many outputs I want per cell?
- Each cell should have: scales_number=3 * bboxes_per_scale_number=1 * (P x y w h c1 c2)=7  -> 21 outputs_per_cell

The final network output must looks like: cells_number (eg, 13x13=169) * outputs_per_cell (eg,21). If 13x13x21 -> 169x21 -> 169 cells with 21 outputs each. 
