package com.mdp.app;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class MapWriter{
    public enum Direction {
        UP, DOWN, LEFT, RIGHT
    }

    private BufferedImage upBufferedImage = null;
    private BufferedImage downBufferedImage = null;
    private BufferedImage leftBufferedImage = null;
    private BufferedImage rightBufferedImage = null;

    private int spriteSize = 8;
    
    MapWriter() {
        try { 
            upBufferedImage = ImageIO.read(new File("assets/up.png"));
            downBufferedImage = ImageIO.read(new File("assets/down.png"));
            leftBufferedImage = ImageIO.read(new File("assets/left.png"));
            rightBufferedImage = ImageIO.read(new File("assets/right.png"));
        } catch(IOException e) {
            System.exit(-1);
        }
    }

    public void write(Direction[][] map, String filename) {
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

    private static void blitImage(BufferedImage in, BufferedImage out, int x, int y) {
        for (int xx = 0;xx < in.getWidth();xx++) {
            for (int yy = 0;yy < in.getWidth();yy++) {
                out.setRGB(xx + x, yy + y, in.getRGB(xx, yy));
            }
        }
    }
};