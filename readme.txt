conda create -n mario
conda activate mario
conda install python=3.8
pip install setuptools==65.5.0
pip install --user wheel==0.38.0 
pip install --upgrade pip wheel==0.38.4
pip install stable-baselines3==1.5
pip install gym-super-mario-bros==7.4.0
pip install opencv-python==
pip install scikit-image
pip3 install tensorboard
pip install memory-profiler psutil

# run ppo
python ./ppo_sb3.py
  run train() for training 
  run test("./checkpoints/....") for testing, replace the path with the path of the checkpoint you want to test 
# run dqn
python ./dqn.py
    run train() for training 
    run test("./checkpoints/....") for testing, replace the path with the path of the checkpoint you want to test
