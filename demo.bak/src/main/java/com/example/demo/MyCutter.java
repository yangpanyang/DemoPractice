package com.example.demo;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.stereotype.Component;

@Component
@Aspect
public class MyCutter {
    // 格式为 包名.类名.函数名(参数)，"*"通配符，".."匹配任意参数
    // 还可以用target/within处理其它情况
    @Pointcut("execution(* com.example.demo.Calc.div(int, int))")
    public void div() {} // 这个函数的作用类似给切点起个名字

    @Before("div()") // 函数执行前，也可以用pointcut指定切面。在around内被调用
    public void before(JoinPoint joinPoint) {
        Object[] args = joinPoint.getArgs();
        System.out.printf("Before %s: %d, %d\n", joinPoint.getSignature().getName(), args[0], args[1]);
    }

    @Around("div()")
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
        System.out.printf("Entering %s\n", pjp.getSignature().getName());
        Object result = pjp.proceed();
        System.out.printf("Leaving %s\n", pjp.getSignature().getName());
        return result;
    }

    @AfterReturning(pointcut = "div()", returning = "ret") //在after之后执行，绑定返回值
    public void afterReturning(Object ret) {
        System.out.printf("AfterReturning, return value is %s\n", ret.toString());
    }

    @After(value="div()") // 在around内被调用，around之后执行
    public void after(JoinPoint joinPoint) {
        System.out.printf("After %s\n", joinPoint.getSignature().getName());
    }

    @AfterThrowing(pointcut = "div()", throwing = "e") //处理异常，需要的话重新包装一下再扔出，默认扔出获得的异常
    public void afterThrowing(JoinPoint joinPoint, Exception e) throws Throwable {
        System.out.printf("AfterThrowing %s: %s\n", joinPoint.getSignature().getName(), e.getMessage());
        // throws new Exception("Fuck"); // 看下此时的异常信息
    }
}
