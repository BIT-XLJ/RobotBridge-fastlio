# RobotBridge
The implementation of sim2sim and sim2real code for humanoid robots; Easy Plug and Use

## Transition Layer Initialization
1. clone this repo to unitree robot's local device or you may use tools like scp to upload the directory. Compiler is required for compiling the code.

2. Then switch to the `unitree_sdk2` directory and do:
    ```Shell
    find . -exec touch -c {} \; # modify the timestamp of files
    mkdir build
    cd build
    cmake ..
    make
    ```
    This will give an executable file named `trans` or `trans_wo_lock` under the `bin` directory.

3. Check the network interface and run the executable file
    ```Shell
    cd bin
    ifconfig # this is used to check the network interface; you may use the name(eth0 or eth1) corresponding to ip address 192.168.123.164
    ./trans_wo_lock eth0
    ```
    This will launch the transition layer. You may start ENTER on keyboard to trigger communication.

## Policy Layer Initialization
1. Prepare the conda environment
    ```Shell
    conda create -n deploy python=3.8
    conda activate deploy
    pip install -r requirements.txt
    ```

2. Test the sim2sim effect
    ```Shell
    cd deploy
    python run.py \
    sim=mujoco \
    device=XXX \
    obs=XXX \
    agent.config.checkpoint=XXX \
    env.config.motion.motion_file=XXX \
    env.config.motion.align_method=XXX \
    sim.config.marker=True \
    sim.config.random_init_heading=True
    ```
    **Device** determines where to place the motionlib and policy model. If you have GPUs, recommend using cuda! Default to cpu!!

    You should first add your observation config to the `config/obs` directory. The config should be organized in the same as current examples!

    Then you should replace the `agent.config.checkpoint` with the path to your checkpoint

    You may also replace the `env.config.motion.motion_file` with the desired motion file to evaluate. Now we only support motion file with only one motion. Otherwise, there may be some unknown behaviors such as random sampling. Recommend manually handling the motion file before using it.

    You may also assign the `env.config.motion.align_method` for aligning the coordinates of reference motion and current robot state. We provide two ways for alignment. One is to align the motion on all directions and the other is to align only on xy-plane. We highly recommend you knowing what these mean before changing this option. Default to xy.

    We also provide you with a visualization tool, `sim.config.marker`. It defaults to True. If you would like some pure visualization effect, you may turn it off.
    
    Most important thing, we provide you with a config named `sim.config.random_init_heading` which will exert a perturbation on the initial heading direction of robots. This is to test the robustness of humanoid robots, as the gravity depends on the root rotation which is a global variable. So, the policy may not be robust enough to handle situations facing different directions. We highly recommend you turning it on and we default it to True.

    You may need to export the following lib for successful running!:
    ```Shell
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    ```

3. Deploy on Real Robot
    ```Shell
    cd deploy
    python run.py \
    sim=real_world
    device=XXX \
    obs=XXX \
    env.config.motion.motion_file=XXX \
    agent.config.checkpoint=XXX \
    env.config.motion.align_method=XXX
    ```
    All the options may follow the same as the sim2sim.
    
    You should first launch the transition layer before initializing the Python script!!!