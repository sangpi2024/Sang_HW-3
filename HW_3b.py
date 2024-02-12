from math import sqrt, pi, gamma

def t_distribution_pdf(t, v):
    """
    Calculate the probability density function of the t-distribution.

    Parameters:
    - t: The t-value, a float.
    - v: Degrees of freedom, an integer.

    Returns:
    - The value of the PDF at t for v degrees of freedom.
    """
    numerator = gamma((v + 1) / 2)
    denominator = sqrt(v * pi) * gamma(v / 2)
    return numerator / denominator * (1 + t**2 / v) ** (-(v + 1) / 2)

def simpsons_rule(a, b, n, v):
    """
    Numerically integrate the t-distribution PDF from a to b using Simpson's rule.

    Parameters:
    - a: The start of the interval, a float.
    - b: The end of the interval, a float.
    - n: Number of intervals, an even integer.
    - v: Degrees of freedom for the t-distribution, an integer.

    Returns:
    - The approximate integral of the PDF over [a, b].
    """
    h = (b - a) / n  # Calculate the width of each interval
    sum1 = 0  # Initialize sum for odd indices
    sum2 = 0  # Initialize sum for even indices

    # Sum contributions of midpoints and endpoints
    for i in range(1, n, 2):
        x = a + i * h
        sum1 += t_distribution_pdf(x, v)
    for i in range(2, n-1, 2):
        x = a + i * h
        sum2 += t_distribution_pdf(x, v)

    # Apply Simpson's rule formula
    return (h/3) * (t_distribution_pdf(a, v) + 4 * sum1 + 2 * sum2 + t_distribution_pdf(b, v))

def main():
    """
    Main function to prompt user input and calculate the probability using Simpson's rule.
    """
    # Prompt user for degrees of freedom and z value
    v = int(input("Enter degrees of freedom (v): "))
    z = float(input("Enter z value: "))
    n = 1000  # Suggested number of intervals for accuracy

    # Compute the probability using Simpson's Rule
    probability = simpsons_rule(0, z, n, v)

    print(f"Probability for v={v} and z={z}: {probability}")

if __name__ == "__main__":
    main()
