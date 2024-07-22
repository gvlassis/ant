script_path="$(readlink -f "${0}")"
src_path="$(dirname "${script_path}")"
root_path="$(dirname "${src_path}")"
out_path="${root_path}/out"

zetas=$(ls $out_path/$dataset/$family/$parametrization | grep "ζ=*" | cut -d = -f 2 | sort -n | tr "\n" " ")
zetas=($zetas)

for zeta in "${zetas[@]}"
do  
    printf "💃ζ=$zeta\n"
    python $src_path/summary.py $out_path/$dataset/$family/$parametrization/ζ=$zeta
done
