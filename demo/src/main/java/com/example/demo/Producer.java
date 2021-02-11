package com.example.demo;

import java.util.Queue;

public class Producer extends Thread {
    String name;  // 生产者的名字
    Queue<String> queue;  // 消息队列

    Producer(String name, Queue<String> queue) {
        this.name = name;
        this.queue = queue;
    }

    public void produce() {
        synchronized (queue) {
            for (int i = 0; i < 5; ++i) {
                queue.add(name + "-" + i);
            }
        }
    }

    @Override
    public void run() {
        produce();
    }
}
