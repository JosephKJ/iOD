# Base 19
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/base_19.yaml
# 19 + 1
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/19_p_1.yaml
# 19 + 1 _ ft
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/ft_19_p_1.yaml


# Base 15
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/base_15.yaml
# 15 + 5
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/15_p_5.yaml
# 15 + 5 _ ft
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/ft_15_p_5.yaml


# Base 10
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/base_10.yaml
# 10 + 10
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/10_p_10.yaml
# 10 + 10 _ ft
sleep 10
python tools/train_net.py --config-file ./configs/PascalVOC-Detection/iOD/ft_10_p_10.yaml