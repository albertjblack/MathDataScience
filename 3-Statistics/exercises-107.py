# 1. spool of 1.75
print(1)
spool_measurements = [1.78, 1.75, 1.72, 1.74, 1.77]
spool_x_bar = sum(spool_measurements) / len(spool_measurements)
spool_measurements_variance = sum(
    [(x - spool_x_bar) ** 2 for x in spool_measurements]
) / (len(spool_measurements) - 1)
spool_measurements_stdev = spool_measurements_variance ** (1 / 2)

print(spool_measurements_stdev)
print()

# 2. z-phone shelf life 42 mths, stdev 8 mths, assume normal dist. ?? prob will last 20 to 30
print(2)
from scipy.stats import norm

z_phone_mean = 42
z_phone_stdev = 8
probability_between_20_and_30 = norm.cdf(30, z_phone_mean, z_phone_stdev) - norm.cdf(
    20, z_phone_mean, z_phone_stdev
)
print(probability_between_20_and_30)
print()

# 3. skeptical that 1.75 mm filament is not 1.75 in avg diameter as advertised. 34 samples gave x_bar of 1.715588, stdev 0.029252 ? what is the 99% confidence interval
print(3)
fil_n_samples = 34
fil_x_bar = 1.715588
fil_stdev = 0.029252
fil_percentage_for_confidence = 0.99

fil_left_tail_area = (1 - fil_percentage_for_confidence) / 2
fil_right_tail_area = 1 - ((1 - fil_percentage_for_confidence) / 2)
fil_norm_dist = norm(loc=0, scale=1)
fil_left_critical_z = fil_norm_dist.ppf(fil_left_tail_area)
fil_right_critical_z = fil_norm_dist.ppf(fil_right_tail_area)

fil_E = (
    fil_left_critical_z * (fil_stdev / fil_n_samples**0.5),
    fil_right_critical_z * (fil_stdev / fil_n_samples**0.5),
)

print(f"99& confidence interval: [{fil_x_bar+fil_E[0]}, {fil_x_bar+fil_E[1]}]")
print()

# 4. know if marketing campaoign affected sales, past generated 10,345 per day with sted of 552. new campaign ran for 45 days and averaged 11,641 in sales
# did the campaign affect sales, yes or no?
print(4)
mean = 10345
std_dev = 552
p1 = 1 - norm.cdf(11641, mean, std_dev)
p2 = p1
p_t = p1 + p2
print(f"better: {p_t<0.05}")
