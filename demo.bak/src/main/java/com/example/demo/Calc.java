package com.example.demo;

import org.springframework.stereotype.Component;

// 可以通过Autowired自动展开，不用显示的实力话这个类
@Component
public class Calc {
    public int div(int x, int y) {
        return x / y;
    }
}
