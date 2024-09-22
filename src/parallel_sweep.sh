script_path="$(readlink -f "${0}")"
src_path="$(dirname "${script_path}")"
root_path="$(dirname "${src_path}")"
out_path="${root_path}/out"

zeta=6
for device_index in {0..3}
do 
    for sweep in {0}
    do
        command="$src_path/sweep.py $out_path/$dataset/$family/$parametrization/ζ=$zeta $* --dataset $dataset --family $family --parametrization $parametrization --ζ $zeta --model_device_index $device_index"

        printf "$command\n"
        python $command &
    done
done

wait
