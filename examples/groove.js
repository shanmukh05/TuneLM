// An original starter composition; see examples/seeds.jsonl for atomic voices.
setcpm(120/4);
const drums = s("bd*4, [~ sd]*2, hh*8").bank("RolandTR909").gain(0.4);
const bass = n("<[0 ~ 0 3] [0 ~ 5 7]>").scale("C2:minor").s("sawtooth").lpf(900).gain(0.4);
const chords = n("<[0,2,4] [3,5,7]>/2").scale("C4:minor").s("triangle").hpf(200).gain(0.2);
stack(drums, bass, chords)
