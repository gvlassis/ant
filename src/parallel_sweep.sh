script_path="$(readlink -f "${0}")"
src_path="$(dirname "${script_path}")"
root_path="$(dirname "${src_path}")"
out_path="${root_path}/out"

zeta=16
for device_index in {0,1,2,3}
do 
    for sweep in {0}
    do
        python $src_path/sweep.py $out_path/$dataset/$family/$parametrization/ζ=$zeta $@ --dataset $dataset --family $family --parametrization $parametrization --ζ $zeta --device_index $device_index &
    done
done
zeta=32
for device_index in {4,5,6,7}
do 
    for sweep in {0}
    do
        python $src_path/sweep.py $out_path/$dataset/$family/$parametrization/ζ=$zeta $@ --dataset $dataset --family $family --parametrization $parametrization --ζ $zeta --device_index $device_index &
    done
done

wait
