package com.mdp.app;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.TerminalFunction;
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
        {-0.04, -0.04,  -0.04,  -0.04},
        {-0.04, -1.0,   -0.04,  -1.0},
        {-0.04, -1.0,   -0.04,  -1.0},
        {-0.04, -0.04,  -0.04,   1.0},
    };

    private enum Direction {
        UP, DOWN, LEFT, RIGHT
    }

    private static BufferedImage upBufferedImage = null;
    private static BufferedImage downBufferedImage = null;
    private static BufferedImage leftBufferedImage = null;
    private static BufferedImage rightBufferedImage = null;

    private static int spriteSize = 8;

    public static void init() {
        try { 
            upBufferedImage = ImageIO.read(new File("assets/up.png"));
            downBufferedImage = ImageIO.read(new File("assets/down.png"));
            leftBufferedImage = ImageIO.read(new File("assets/left.png"));
            rightBufferedImage = ImageIO.read(new File("assets/right.png"));
        } catch(IOException e) {
            System.exit(-1);
        }
    }

    /**
     * Main function
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
        init();

        int height = map.length;
        int width = map[0].length;
        
        GridWorldDomain gw = new GridWorldDomain(width, height);
        gw.setProbSucceedTransitionDynamics(0.9);

        GridWorldRewardFunction rf = new GridWorldRewardFunction(width, height);
        for (int y = 0;y < height;y++) {
            for (int x = 0; x < width; x++) {
                rf.setReward(x, y, map[y][x]);
            }
        }
        TerminalFunction tf = new GridWorldTerminalFunction(3, 3);

        gw.setRf(rf);
        gw.setTf(tf);
        
        OOSADomain domain = gw.generateDomain();
        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        GridWorldState state = new GridWorldState(new GridAgent(0, 0, "Start"));

        ValueIteration vi = new ValueIteration(domain, 0.99, hashingFactory, 0.01, 1000);
        GreedyQPolicy viPolicy = vi.planFromState(state);

        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.01, 1000, 1000);
        GreedyQPolicy piPolicy = pi.planFromState(state);

        vi.toggleDebugPrinting(true);

        Direction[][] viDirections = new Direction[height][width];
        Direction[][] piDirections = new Direction[height][width];

        for (int y = height - 1;y >= 0;y--) {
            for (int x = 0; x < width; x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (viPolicy.action(currentState).actionName() == "east") {
                    viDirections[y][x] = Direction.RIGHT;
                } else if (viPolicy.action(currentState).actionName() == "west") {
                    viDirections[y][x] = Direction.LEFT;
                } else if (viPolicy.action(currentState).actionName() == "north") {
                    viDirections[y][x] = Direction.UP;
                } else if (viPolicy.action(currentState).actionName() == "south") {
                    viDirections[y][x] = Direction.DOWN;
                }
            }
        }

        for (int y = height - 1;y >= 0;y--) {
            for (int x = 0; x < width; x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (piPolicy.action(currentState).actionName() == "east") {
                    piDirections[y][x] = Direction.RIGHT;
                } else if (piPolicy.action(currentState).actionName() == "west") {
                    piDirections[y][x] = Direction.LEFT;
                } else if (piPolicy.action(currentState).actionName() == "north") {
                    piDirections[y][x] = Direction.UP;
                } else if (piPolicy.action(currentState).actionName() == "south") {
                    piDirections[y][x] = Direction.DOWN;
                }
            }
        }

        writeOutputActionMap(viDirections, "output/vi.png");
        writeOutputActionMap(piDirections, "output/pi.png");
    }

    public static void writeOutputActionMap(Direction[][] map, String filename) {
        int width = map[0].length;
        int height = map.length;


        BufferedImage outputImage = new BufferedImage(width * spriteSize, height * spriteSize, BufferedImage.TYPE_4BYTE_ABGR);
        for (int y = 0;y < height;++y) {
            for (int x = 0;x < width;++x) {
                if (map[height - 1 - y][x] == Direction.UP) {
                    blitImage(upBufferedImage, outputImage, x * spriteSize, y * spriteSize);
                } else if (map[height - 1 - y][x] == Direction.DOWN) {
                    blitImage(downBufferedImage, outputImage, x * spriteSize, y * spriteSize);
                } else if (map[height - 1 - y][x] == Direction.LEFT) {
                    blitImage(leftBufferedImage, outputImage, x * spriteSize, y * spriteSize);
                } else if (map[height - 1 - y][x] == Direction.RIGHT) {
                    blitImage(rightBufferedImage, outputImage, x * spriteSize, y * spriteSize);
                }
            }
        }
        try {
            ImageIO.write(outputImage, "png", new File(filename));
        } catch(IOException e) {
            System.exit(-1);
        }
    }

    public static void blitImage(BufferedImage in, BufferedImage out, int x, int y) {
        for (int xx = 0;xx < in.getWidth();xx++) {
            for (int yy = 0;yy < in.getWidth();yy++) {
                out.setRGB(xx + x, yy + y, in.getRGB(xx, yy));
            }
        }
    }
}
