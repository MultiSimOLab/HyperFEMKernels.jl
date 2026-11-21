# Groups of cases of the performance gate and the group of each case identifier (prefix before "/"): shared by
# cases.jl (in the measuring processes) and driver.jl (which does not load the packages).
const GROUP_NAMES = ["sims", "em", "visco", "forms", "pattern", "aorta_p1p1", "aorta_p2p1", "hmre_vacuum_2d"]
const QUICK_GROUPS = ["sims", "em", "pattern"]
const PREFIX_GROUP = Dict("sims" => "sims", "em_box_q2" => "em", "em_plate_p1" => "em", "visco" => "visco", "forms" => "forms",
                          "pattern" => "pattern", "aorta_p1p1" => "aorta_p1p1", "aorta_p2p1" => "aorta_p2p1",
                          "hmre_vacuum_2d" => "hmre_vacuum_2d")
group_of(id) = get(PREFIX_GROUP, first(split(id, '/')), "")
