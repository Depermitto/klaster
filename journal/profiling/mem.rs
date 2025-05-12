fn process_strings(count: usize) -> Vec<String> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let s = " ".repeat(i * 1000);
        result.push(s);
    }
    result
}

fn main() {
    for _ in 0..100 {
        let strings = process_strings(100);
		_ = strings;
    }
}
