package com.mdp.app;

import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

/**
 * Hello world!
 */
public final class App {
    private App() {
    }

    public static final int WIDTH = 4;
    public static final int HEIGHT = 4;

    public static final double[][] map = new double[][]{
        {0.0,   1.0,    1.0,   -1.0},
        {1.0,  -1.0,    1.0,   -1.0},
        {1.0,  -1.0,    1.0,   -1.0},
        {1.0,   1.0,    1.0,    0.0},
    };

    /**
     * Main function
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
        int height = map.length;
        int width = map[0].length;

        GridWorldDomain gw = new GridWorldDomain(width, height);

        GridWorldRewardFunction rf = new GridWorldRewardFunction(width, height);
        for (int y = 0;y < height;y++) {
            for (int x = 0; x < width; x++) {
                rf.setReward(x, y, map[x][y]);
            }
        }
        TerminalFunction tf = new GridWorldTerminalFunction(3, 3);

        gw.setRf(rf);
        gw.setTf(tf);
        
        OOSADomain domain = gw.generateDomain();

        GridWorldState state = new GridWorldState(
            new GridAgent(0, 0, "Start"),
            new GridLocation(3, 3, "Goal"));
        
        ConstantStateGenerator sg = new ConstantStateGenerator(state);
        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        LearningAgentFactory qLearningFactory = new LearningAgentFactory(){
        
            @Override
            public String getAgentName() {
                return "Q-Learning";
            }
        
            @Override
            public LearningAgent generateAgent() {
                return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
            }
        };

        SimulatedEnvironment environment = new SimulatedEnvironment(domain, sg);
        LearningAlgorithmExperimenter experimenter = new LearningAlgorithmExperimenter(
            environment, 10, 100, qLearningFactory);
        
        experimenter.setUpPlottingConfiguration(
            500, 250, 2, 1000,
            TrialMode.MOST_RECENT_AND_AVERAGE,
            PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
            PerformanceMetric.AVERAGE_EPISODE_REWARD);

        experimenter.startExperiment();        
    }
}
