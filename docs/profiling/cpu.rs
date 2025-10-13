fn fib(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn a(x: f64, y: f64) -> f64 {
    let mut x = x;
    let mut y = y;
    let mut z = 0.0;
    while x > y {
        x /= 2.0;
        y *= 2.0;
        z += a(x, y);
    }
    z
}

fn b(x: f64, y: f64, z: f64) -> f64 {
    a(x, y) + a(y, z) * a(z, x)
}

fn c(x: i32) -> i32 {
    if x < 0 {
        b((x - 1) as f64, x as f64, (x + 1) as f64) as i32
    } else {
        10
    }
}

fn main() {
    let n = 30;
    for i in 0..n {
        let mut x = fib(i);
        x += a(x as f64, i as f64) as i32;
        x -= b(i as f64, x as f64, (i + x) as f64) as i32;
        x *= c(i);
        _ = x;
    }
}
