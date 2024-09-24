#![forbid(unsafe_code)]
#![cfg_attr(feature = "no-std", no_std)]

// #![warn(missing_docs)]
// #![warn(missing_doc_code_examples)]

#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use arrayvec::ArrayVec;

#[cfg(feature = "std")]
type Num = f64;
#[cfg(feature = "no-std")]
type Num = Decimal;

#[cfg(feature = "std")]
const E: Num = std::f64::consts::E;
#[cfg(feature = "std")]
const ZERO: Num = 0.0;
#[cfg(feature = "std")]
const ONE: Num = 1.0;
#[cfg(feature = "std")]
const TWO: Num = 2.0;

#[cfg(feature = "no-std")]
const E: Num = Decimal::E;
#[cfg(feature = "no-std")]
const ZERO: Num = Decimal::ZERO;
#[cfg(feature = "no-std")]
const ONE: Num = Decimal::ONE;
#[cfg(feature = "no-std")]
const TWO: Num = Decimal::TWO;

#[cfg(feature = "no-std")]
use rust_decimal::prelude::*;

#[derive(Debug, Default, Clone, Copy)]
enum Outlier {
    Mild(Num),
    Extreme(Num),
    #[default]
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct Outliers<const N: usize>([Outlier; N]);

impl<const N: usize> core::fmt::Display for Outliers<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Outliers(outliers) = self;

        let extremes = outliers
            .iter()
            .filter_map(|outlier| match outlier {
                Outlier::Extreme(e) => Some(e),
                _ => None,
            })
            .collect::<Vec<_>>();

        let milds = outliers
            .iter()
            .filter_map(|outlier| match outlier {
                Outlier::Mild(m) => Some(m),
                _ => None,
            })
            .collect::<Vec<_>>();

        write!(f, "Extreme Outliers: {extremes:?}; Mild Outliers: {milds:?}")
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Description<'a> {
    count: usize,
    mean: Num,
    std: Num,
    max: Num,
    min: Num,
    nan_allowed: bool,
    description_percentiles: &'a [DescriptionPercentile],
}

impl<'a> core::fmt::Display for Description<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "count\t{}", self.count)?;
        writeln!(f, "mean\t{}", self.mean)?;
        writeln!(f, "std\t{}", self.std)?;
        writeln!(f, "max\t{}", self.max)?;
        writeln!(f, "min\t{}", self.min)?;

        if self.nan_allowed {
            writeln!(f, "*NaN:\tAllowed*")?;
        }

        for dp in self.description_percentiles {
            writeln!(f, "{dp}")?;
        }

        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DescriptionPercentile {
    percentile: Num,
    value: Num,
}

impl core::fmt::Display for DescriptionPercentile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}\t{}", self.percentile, self.value)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Quartiles(Num, Num, Num);

impl core::fmt::Display for Quartiles {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Quartiles(q1, q2, q3) = self;
        write!(f, "(q1: {q1}, q2: {q2}, q3: {q3})")
    }
}

#[derive(Debug, Default, Clone)]
pub struct Series<const N: usize>(ArrayVec<Num, N>);

// Conversion traits
impl<const N: usize, T: Copy + Into<Num>> From<&[T]> for Series<N> {
    fn from(value: &[T]) -> Self {
        Self(value.into_iter().map(|x| Into::into(*x)).collect())
    }
}

impl<const N: usize, T: Copy + Into<Num>> From<[T; N]> for Series<N> {
    fn from(value: [T; N]) -> Self {
        Self(value.into_iter().map(Into::into).collect())
    }
}

impl<const N: usize, A: Into<Num>> FromIterator<A> for Series<N> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl<const N: usize> AsRef<ArrayVec<Num, N>> for Series<N> {
    fn as_ref(&self) -> &ArrayVec<Num, N> {
        self.inner()
    }
}

// Display trait
impl<const N: usize> core::fmt::Display for Series<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_empty() {
            write!(f, "Series{{}}")
        } else {
            write!(f, "Series{:?}", self.0)
        }
    }
}

// Iterator trait
impl<'a, const N: usize> IntoIterator for &'a Series<N> {
    type Item = <core::slice::Iter<'a, Num> as Iterator>::Item;

    type IntoIter = core::slice::Iter<'a, Num>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner().as_slice().into_iter()
    }
}

// Ops trait

impl<const N: usize> core::ops::Deref for Series<N> {
    type Target = [Num];

    fn deref(&self) -> &Self::Target {
        &self.inner().as_slice()[..N]
    }
}

impl<I: core::slice::SliceIndex<[Num]>, const N: usize> core::ops::Index<I>
    for Series<N>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.inner()[index]
    }
}

impl<const N: usize> core::ops::Add for Series<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect())
    }
}

impl<const N: usize> core::ops::Sub for Series<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.iter().zip(rhs.iter()).map(|(l, r)| l - r).collect())
    }
}

impl<const N: usize> core::ops::Mul for Series<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.iter().zip(rhs.iter()).map(|(l, r)| l * r).collect())
    }
}

impl<const N: usize> core::ops::Div for Series<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.iter().zip(rhs.iter()).map(|(l, r)| l / r).collect())
    }
}

impl<const N: usize> core::ops::Neg for Series<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.iter().map(core::ops::Neg::neg).collect())
    }
}

impl<const N: usize> core::ops::Rem for Series<N> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.iter().zip(rhs.iter()).map(|(l, r)| l % r).collect())
    }
}

// Cmp traits
impl<const N: usize> PartialEq for Series<N> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<const N: usize> Eq for Series<N> {}

// Utility Functions
impl<const N: usize> Series<N> {
    #[inline(always)]
    fn inner(&self) -> &ArrayVec<Num, N> {
        &self.0
    }

    #[inline(always)]
    const fn is_empty(&self) -> bool {
        N == 0
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn casted_len(&self) -> Num {
        N as Num
    }

    #[cfg(feature = "no-std")]
    #[inline(always)]
    fn casted_len(&self) -> Num {
        Num::from(N)
    }
}

// User-end functions
impl<const N: usize> Series<N> {
    pub fn new<T: Copy + TryInto<Num>>(data: &[T; N]) -> Option<Self> {
        let data = data
            .iter()
            .map(|x| (*x).try_into().ok())
            .collect::<Option<ArrayVec<_, N>>>()?;

        Some(Self(data))
    }

    #[cfg(feature = "std")]
    #[inline]
    pub fn sorted(&self) -> Self {
        let Series(mut series) = self.clone();
        series.sort_by(|a, b| cmp_floats(&a, &b));
        Self(series)
    }

    #[cfg(feature = "no-std")]
    #[inline]
    pub fn sorted(&self) -> Self {
        let Series(mut series) = self.clone();
        series.sort();
        Self(series)
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&Num> {
        self.inner().get(index)
    }

    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    #[inline]
    pub fn less(&self, i: usize, j: usize) -> bool {
        self[i] < self[j]
    }

    #[inline]
    pub fn swap(self, i: usize, j: usize) -> Self {
        let Self(mut series) = self;
        series.swap(i, j);
        Self(series)
    }
    #[cfg(feature = "std")]
    #[inline]
    pub fn min(&self) -> Option<&Num> {
        self.iter().min_by(cmp_floats)
    }

    #[cfg(feature = "no-std")]
    #[inline]
    pub fn min(&self) -> Option<&Num> {
        self.iter().min()
    }

    #[cfg(feature = "std")]
    #[inline]
    pub fn max(&self) -> Option<&Num> {
        self.iter().max_by(cmp_floats)
    }

    #[cfg(feature = "no-std")]
    #[inline]
    pub fn max(&self) -> Option<&Num> {
        self.iter().max()
    }

    #[inline]
    pub fn sum(&self) -> Num {
        self.iter().sum()
    }
}

// Statistical Operations
impl<const N: usize> Series<N> {
    #[inline]
    pub fn cumulative_sum(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            let mut cummulative_sum = ArrayVec::<Num, N>::new();

            for (i, val) in self.iter().enumerate() {
                if i == 0 {
                    cummulative_sum[i] = *val;
                } else {
                    cummulative_sum[i] = cummulative_sum[i - 1] + *val;
                }
            }

            Some(Self(cummulative_sum))
        }
    }

    #[cfg(feature = "std")]
    #[inline]
    pub fn mean(&self) -> Option<Num> {
        (!self.is_empty()).then_some(self.sum() / self.casted_len())
    }

    #[cfg(feature = "no-std")]
    #[inline]
    pub fn mean(&self) -> Option<Num> {
        (!self.is_empty()).then_some(self.sum() / self.casted_len())
    }

    #[inline]
    pub fn median(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let Series(series) = self.sorted();
            Some(median(&series))
        }
    }

    pub fn mode(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let mut counter = 0;
            let mut curr: Option<Num> = None;

            for i in 0..N {
                let mut count = 0;
                for j in 0..N {
                    if self[i] == self[j] {
                        count += 1;
                    }
                }
                if count > counter {
                    counter = count;
                    curr = Some(self[i]);
                }
            }

            curr
        }
    }

    #[inline]
    pub fn geometric_mean(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let p = self.iter().product::<Num>();

            Some(p.powf(1.0 / N as f64))
        }
    }

    pub fn harmonic_mean(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let p = self.iter().fold(ZERO, |mut acc, d| {
                acc += ONE / *d;
                acc
            });

            Some(self.casted_len() / p)
        }
    }

    pub fn median_absolute_deviation_population(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let m = self.median()?;

            let series = Self(self.iter().map(|d| (d - m).abs()).collect());

            series.median()
        }
    }

    #[cfg(feature = "std")]
    pub fn standard_deviation(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(self, 0).sqrt())
        }
    }

    #[cfg(feature = "no-std")]
    pub fn standard_deviation(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            variance(self, 0).sqrt()
        }
    }

    #[cfg(feature = "std")]
    pub fn standard_deviation_population(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(self, 0).sqrt())
        }
    }

    #[cfg(feature = "no-std")]
    pub fn standard_deviation_population(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            variance(self, 0).sqrt()
        }
    }

    #[cfg(feature = "std")]
    pub fn standard_deviation_sample(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(self, 1).sqrt())
        }
    }

    #[cfg(feature = "no-std")]
    pub fn standard_deviation_sample(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            variance(self, 1).sqrt()
        }
    }

    pub fn quartile_outliers(&self) -> Option<Outliers<N>> {
        if self.is_empty() {
            None
        } else {
            let series = self.sorted();
            let Quartiles(q1, _q2, q3) = series.quartile()?;
            let iqr = series.inter_quartile_range()?;

            #[cfg(feature = "std")]
            let in_fence = 1.5 * iqr;
            #[cfg(feature = "std")]
            let out_fence = 3.0 * iqr;

            #[cfg(feature = "no-std")]
            let in_fence = Decimal::new(15, 1) * iqr;
            #[cfg(feature = "no-std")]
            let out_fence = Decimal::new(30, 1) * iqr;

            let lif = q1 - in_fence;
            let uif = q3 + in_fence;

            let lof = q1 - out_fence;
            let uof = q3 + out_fence;

            let mut outliers = [Outlier::default(); N];

            for (i, d) in series.iter().enumerate() {
                if d < &lof || d > &uof {
                    outliers[i] = Outlier::Extreme(*d);
                } else if d < &lif || d > &uif {
                    outliers[i] = Outlier::Mild(*d);
                } else {
                    outliers[i] = Outlier::None;
                }
            }

            Some(Outliers(outliers))
        }
    }

    pub fn percentile(&self, percent: f64) -> Option<Num> {
        if self.is_empty() || (percent < 0.0 || percent > 100.0) {
            None
        } else {
            todo!()
        }
    }

    pub fn correlation(&self, other: &Self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let sd1 = self.standard_deviation_population()?;
            let sd2 = other.standard_deviation_population()?;

            if sd1 == ZERO || sd2 == ZERO {
                None
            } else {
                let cov = self.covariance(other)?;

                Some(cov / (sd1 * sd2))
            }
        }
    }

    #[cfg(feature = "std")]
    pub fn auto_correlation(&self, lags: usize) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let mean = self.mean()?;

            let (mut result, mut q) = (ZERO, ZERO);

            for _ in 0..lags {
                let mut v = (self[0] - mean) * (self[0] - mean);
                for i in 1..N {
                    let delta0 = self[i - 1] - mean;
                    let delta1 = self[i] - mean;

                    q += (delta0 * delta1 - q) / (i + 1) as Num;
                    v += (delta0 * delta1 - v) / (i + 1) as Num;
                }

                result = q / v;
            }

            Some(result)
        }
    }

    #[cfg(feature = "no-std")]
    pub fn auto_correlation(&self, lags: usize) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let mean = self.mean()?;

            let (mut result, mut q) = (ZERO, ZERO);

            for _ in 0..lags {
                let mut v = (self[0] - mean) * (self[0] - mean);
                for i in 1..N {
                    let delta0 = self[i - 1] - mean;
                    let delta1 = self[i] - mean;

                    q += (delta0 * delta1 - q) / Num::from(i + 1);
                    v += (delta0 * delta1 - v) / Num::from(i + 1);
                }

                result = q / v;
            }

            Some(result)
        }
    }

    pub fn pearson(&self, other: &Self) -> Option<Num> {
        self.correlation(other)
    }

    pub fn quartile(&self) -> Option<Quartiles> {
        if self.is_empty() {
            None
        } else {
            let c = self.sorted();

            let len = self.len();

            let (c1, c2) = if len % 2 == 0 {
                (len / 2, len / 2)
            } else {
                let tmp = (len - 1) / 2;
                (tmp, tmp + 1)
            };

            let q1 = median(&c[..c1]);
            let q2 = median(&c);
            let q3 = median(&c[c2..]);

            Some(Quartiles(q1, q2, q3))
        }
    }

    pub fn inter_quartile_range(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let qs = self.quartile()?;
            let iqr = qs.2 - qs.0;
            Some(iqr)
        }
    }

    pub fn midhinge(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let qs = self.quartile()?;
            let mh = (qs.2 + qs.0) / TWO;
            Some(mh)
        }
    }

    pub fn trimean(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let c = self.sorted();
            let q = c.quartile()?;
            let tm = (q.0 + (q.1 * TWO) + q.2) / (TWO * TWO);
            Some(tm)
        }
    }

    pub fn sample(&self) -> Option<Series<N>> {
        if self.is_empty() {
            None
        } else {
            todo!()
        }
    }

    pub fn variance(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(&self, 0))
        }
    }

    pub fn population_variance(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(&self, 0))
        }
    }

    pub fn sample_variance(&self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            Some(variance(&self, 1))
        }
    }

    #[cfg(feature = "std")]
    pub fn covariance(&self, other: &Self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let (m1, m2) = (self.mean()?, other.mean()?);

            let mut ss = ZERO;

            for (i, (d1, d2)) in self.iter().zip(other.iter()).enumerate() {
                ss += (((d1 - m1) * (d2 - m2)) - ss) / (i + 1) as Num;
            }

            Some(ss * N as Num / (N - 1) as Num)
        }
    }

    #[cfg(feature = "no-std")]
    pub fn covariance(&self, other: &Self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let (m1, m2) = (self.mean()?, other.mean()?);

            let mut ss = ZERO;

            for (i, (d1, d2)) in self.iter().zip(other.iter()).enumerate() {
                ss += (((d1 - m1) * (d2 - m2)) - ss) / Decimal::from(i + 1);
            }

            Some(ss * self.casted_len() / Num::from(N - 1))
        }
    }

    #[cfg(feature = "std")]
    pub fn covariance_population(&self, other: &Self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let (m1, m2) = (self.mean()?, other.mean()?);

            let mut s = ZERO;

            for (i, (d1, d2)) in self.iter().zip(other.iter()).enumerate() {
                s += (((d1 - m1) * (d2 - m2)) - s) / (i + 1) as Num;
            }

            Some(s / self.casted_len())
        }
    }

    #[cfg(feature = "no-std")]
    pub fn covariance_population(&self, other: &Self) -> Option<Num> {
        if self.is_empty() {
            None
        } else {
            let (m1, m2) = (self.mean()?, other.mean()?);

            let mut s = ZERO;

            for (i, (d1, d2)) in self.iter().zip(other.iter()).enumerate() {
                s += (((d1 - m1) * (d2 - m2)) - s) / Decimal::from(i + 1);
            }

            Some(s / self.casted_len())
        }
    }

    pub fn sigmoid(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            Some(Self(
                self.iter()
                    .map(|d| {
                        let f: f64 = (*d).try_into().unwrap();
                        ONE / (ONE + E.powf(-f))
                    })
                    .collect(),
            ))
        }
    }

    #[cfg(feature = "std")]
    pub fn softmax(&self) -> Option<Series<N>> {
        if self.is_empty() {
            None
        } else {
            let mut s = ZERO;

            let c = self.max()?;

            for d in self {
                s += E.powf(d - c);
            }

            Some(Self(self.iter().map(|x| E.powf(x - c) / s).collect()))
        }
    }

    #[cfg(feature = "no-std")]
    pub fn softmax(&self) -> Option<Series<N>> {
        if self.is_empty() {
            None
        } else {
            let mut s = ZERO;

            let c = self.max()?;

            for d in self {
                s += E.powd(d - c);
            }

            Some(Self(self.iter().map(|x| E.powd(x - c) / s).collect()))
        }
    }

    pub fn entropy(&self) -> Num {
        let ns = normalize::<N>(&self.inner());

        let mut result = ZERO;

        for s in ns {
            if s == ZERO {
                continue;
            } else {
                result += s * log2(&s);
            }
        }

        -result
    }
}

fn median(series: &[Num]) -> Num {
    let len = series.len();

    if len % 2 == 0 {
        let (l, r) = (series[len / 2 - 1], series[len / 2 + 1]);

        (l + r) / TWO
    } else {
        series[len / 2]
    }
}

#[cfg(feature = "std")]
#[inline(always)]
fn cmp_floats(a: &&Num, b: &&Num) -> core::cmp::Ordering {
    a.total_cmp(b)
}

#[cfg(feature = "std")]
#[inline(always)]
fn log2(num: &Num) -> Num {
    num.log2()
}

#[cfg(feature = "no-std")]
#[inline(always)]
fn log2(num: &Num) -> Num {
    num.log10() / TWO.log10()
}

#[cfg(feature = "std")]
fn variance(series: &[Num], sample: usize) -> Num {
    let sum = series.iter().sum::<Num>();
    let len = series.len() as Num;

    let mean = sum / len;

    let mut n = ZERO;

    for d in series {
        n += (d - mean) * (mean - d);
    }

    let variance = n / (len - (1 * sample) as Num);

    variance
}

#[cfg(feature = "no-std")]
fn variance(series: &[Num], sample: usize) -> Num {
    let sum = series.iter().sum::<Num>();
    let len = Num::from(series.len());

    let mean = sum / len;

    let mut n = ZERO;

    for d in series {
        n += (d - mean) * (mean - d);
    }

    let variance = n / (len - Num::from(1 * sample));

    variance
}

fn normalize<const N: usize>(series: &[Num]) -> ArrayVec<Num, N> {
    let sum = series.iter().sum::<Num>();

    series.iter().map(|d| d / sum).collect()
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test_std {
    use super::*;

    lazy_static::lazy_static! {
        static ref S1: Series<10> = Series::new(&[
            49.889236601963056,
            27.90710472204815,
            50.55519600554194,
            61.44965605282565,
            32.48984736216341,
            68.84542257759047,
            118.99248854518666,
            112.93389550501415,
            101.65969266660504,
            71.88108740653236
        ])
        .unwrap();
        static ref S2: Series<20> = Series::new(&[
            45.278308509899446,
            78.6073704548718,
            121.93074945684135,
            71.25800637201132,
            123.21554886683052,
            66.85266070821011,
            21.63129835751687,
            97.00361276738151,
            120.49179545581856,
            29.294260977268838,
            113.58808222383439,
            33.20473830011804,
            11.828505499015046,
            41.65167481809596,
            8.478962989938694,
            82.45843927835531,
            80.28984640765721,
            1.452828443496243,
            5.348598365185607,
            102.31546899564323
        ])
        .unwrap();
        static ref S3: Series<45> = Series::new(&[
            119.50486249261775,
            84.13421282126735,
            31.375815613182244,
            2.2077357778363256,
            107.5547782917988,
            100.49808523606251,
            70.6938210002916,
            38.17905202121143,
            31.108385660367613,
            74.33671485969066,
            37.20098232842954,
            36.166988165791025,
            26.34679470725635,
            114.71804807549309,
            95.84778300503109,
            48.32810807014782,
            42.863118203710876,
            6.769191929077202,
            18.293965263063313,
            102.36246576522613,
            5.0902500384730835,
            71.78418193976613,
            44.830254572401984,
            32.141714321200006,
            24.373938376669667,
            2.068537995180982,
            111.74999824410074,
            90.64682823311183,
            92.39218153376363,
            92.85395380337776,
            22.000277551771312,
            70.06041891168162,
            93.87931526009802,
            77.14819439719551,
            11.608898132776837,
            96.87923375042378,
            22.66643583711647,
            47.14367310720676,
            68.69145657528226,
            72.07328560760875,
            65.28470879768561,
            78.5995605333211,
            13.008782215110587,
            107.68471252503748,
            24.99036806911677
        ])
        .unwrap();
        static ref S4: Series<95> = Series::new(&[
            32.011291185829386,
            41.16885944087332,
            40.942858434710736,
            106.10924460916148,
            75.53203299345262,
            73.95149893049506,
            14.872097219497107,
            67.78751630697572,
            82.77231740415732,
            42.7241543519345,
            77.64589193969962,
            35.85791558401924,
            56.153301785667736,
            105.13900149413487,
            74.35031755858905,
            108.05803668120333,
            51.90499484638232,
            68.87966183273122,
            110.49052926071727,
            75.22123829890211,
            107.77010808492001,
            4.607165988546264,
            56.15610291849866,
            99.14605022210742,
            101.49955820924457,
            72.03998325986862,
            31.30568822373622,
            90.78472112158481,
            103.37900554093656,
            47.919508052764094,
            18.237373695991618,
            13.27391860912538,
            29.159433643868773,
            45.06060362988455,
            0.6141632066634193,
            15.422793069829462,
            100.48797673241812,
            89.3576218159534,
            119.01587846209448,
            71.8670875386896,
            117.37847457893182,
            90.60125715586405,
            78.55976325363845,
            90.10002408578609,
            37.91864519496888,
            6.621009961501429,
            11.690967239164777,
            119.9202419119762,
            83.55433658190775,
            62.95903934829104,
            16.343347086671592,
            51.24043631177927,
            70.39110816261291,
            57.631583291713774,
            10.698063492970508,
            92.41569299379307,
            15.38866720150581,
            83.76735225720627,
            12.514868866764784,
            1.5862487888732266,
            32.87819903597594,
            77.69443311609979,
            7.6287902239824295,
            69.30587949890983,
            83.91005839675323,
            43.751888690665886,
            83.77756135448969,
            86.65631034743015,
            99.68028988552462,
            66.00529313906337,
            52.0814234902964,
            81.78960267875445,
            78.69205444821343,
            119.4361655601349,
            37.81275632823978,
            40.408788061787554,
            119.00525900890923,
            61.66297729091732,
            45.52840580615155,
            66.07547267633618,
            39.27152038672825,
            36.528383513130024,
            15.06016278468826,
            91.25218213697748,
            66.53422079067064,
            1.6410848308236559,
            36.80628261169339,
            13.543602205248789,
            45.247503361751555,
            98.90579766450963,
            80.2899059149801,
            61.47596810076844,
            17.06509748484622,
            77.9244140972354,
            101.50016207442759
        ])
        .unwrap();
    }

    fn precision_f64(x: f64, decimals: u32) -> f64 {
        // if x == 0. || decimals == 0 {
        //     0.
        // } else {
        //     let shift = decimals as i32 - x.abs().log10().ceil() as i32;
        //     let shift_factor = 10_f64.powi(shift);

        //     (x * shift_factor).round() / shift_factor
        // }

        let y = 10i64.pow(decimals) as f64;
        (x * y).round() / y
    }

    #[test]
    fn test_mean() {
        let s1_mean = S1.mean().map(|m| precision_f64(m, 4));
        assert_eq!(s1_mean, Some(69.6604));
        let s2_mean = S2.mean().map(|m| precision_f64(m, 4));
        assert_eq!(s2_mean, Some(62.8090));
        let s3_mean = S3.mean().map(|m| precision_f64(m, 4));
        assert_eq!(s3_mean, Some(58.4032));
        let s4_mean = S4.mean().map(|m| precision_f64(m, 4));
        assert_eq!(s4_mean, Some(61.1241));
    }

    #[test]
    fn test_median() {
        let s1_median = S1.median().map(|m| precision_f64(m, 4));
        assert_eq!(s1_median, Some(66.6654));
        let s2_median = S2.median().map(|m| precision_f64(m, 4));
        assert_eq!(s2_median, Some(72.7300));
        let s3_median = S3.median().map(|m| precision_f64(m, 4));
        assert_eq!(s3_median, Some(65.2847));
        let s4_median = S4.median().map(|m| precision_f64(m, 4));
        assert_eq!(s4_median, Some(66.0755));
    }

    #[test]
    fn test_sum() {
        let s1_sum = Some(precision_f64(S1.sum(), 5));
        assert_eq!(s1_sum, Some(696.60363));
        let s2_sum = Some(precision_f64(S2.sum(), 5));
        assert_eq!(s2_sum, Some(1256.18076));
        let s3_sum = Some(precision_f64(S3.sum(), 5));
        assert_eq!(s3_sum, Some(2628.14207));
        let s4_sum = Some(precision_f64(S4.sum(), 5));
        assert_eq!(s4_sum, Some(5806.78653));
    }

    #[test]
    fn test_geometric_mean() {
        let s1_gm = S1.geometric_mean().map(|gm| precision_f64(gm, 5));
        assert_eq!(s1_gm, Some(62.77312));
        let s2_gm = S2.geometric_mean().map(|gm| precision_f64(gm, 5));
        assert_eq!(s2_gm, Some(40.71189));
        let s3_gm = S3.geometric_mean().map(|gm| precision_f64(gm, 5));
        assert_eq!(s3_gm, Some(42.16271));
        let s4_gm = S4.geometric_mean().map(|gm| precision_f64(gm, 5));
        assert_eq!(s4_gm, Some(45.55146));
    }

    #[test]
    fn test_max() {
        assert_eq!(S1.max(), Some(118.99248854518666).as_ref());
        assert_eq!(S2.max(), Some(123.21554886683052).as_ref());
        assert_eq!(S3.max(), Some(119.50486249261775).as_ref());
        assert_eq!(S4.max(), Some(119.9202419119762).as_ref());
    }

    #[test]
    fn test_min() {
        assert_eq!(S1.min(), Some(27.90710472204815).as_ref());
        assert_eq!(S2.min(), Some(1.452828443496243).as_ref());
        assert_eq!(S3.min(), Some(2.068537995180982).as_ref());
        assert_eq!(S4.min(), Some(0.6141632066634193).as_ref());
    }
}

#[cfg(feature = "no-std")]
#[cfg(test)]
mod test_no_std {
    use super::*;

    lazy_static::lazy_static! {
        static ref S1: Series<10> = Series::new(&[
            49.889236601963056,
            27.90710472204815,
            50.55519600554194,
            61.44965605282565,
            32.48984736216341,
            68.84542257759047,
            118.99248854518666,
            112.93389550501415,
            101.65969266660504,
            71.88108740653236
        ])
        .unwrap();
        static ref S2: Series<20> = Series::new(&[
            45.278308509899446,
            78.6073704548718,
            121.93074945684135,
            71.25800637201132,
            123.21554886683052,
            66.85266070821011,
            21.63129835751687,
            97.00361276738151,
            120.49179545581856,
            29.294260977268838,
            113.58808222383439,
            33.20473830011804,
            11.828505499015046,
            41.65167481809596,
            8.478962989938694,
            82.45843927835531,
            80.28984640765721,
            1.452828443496243,
            5.348598365185607,
            102.31546899564323
        ])
        .unwrap();
        static ref S3: Series<45> = Series::new(&[
            119.50486249261775,
            84.13421282126735,
            31.375815613182244,
            2.2077357778363256,
            107.5547782917988,
            100.49808523606251,
            70.6938210002916,
            38.17905202121143,
            31.108385660367613,
            74.33671485969066,
            37.20098232842954,
            36.166988165791025,
            26.34679470725635,
            114.71804807549309,
            95.84778300503109,
            48.32810807014782,
            42.863118203710876,
            6.769191929077202,
            18.293965263063313,
            102.36246576522613,
            5.0902500384730835,
            71.78418193976613,
            44.830254572401984,
            32.141714321200006,
            24.373938376669667,
            2.068537995180982,
            111.74999824410074,
            90.64682823311183,
            92.39218153376363,
            92.85395380337776,
            22.000277551771312,
            70.06041891168162,
            93.87931526009802,
            77.14819439719551,
            11.608898132776837,
            96.87923375042378,
            22.66643583711647,
            47.14367310720676,
            68.69145657528226,
            72.07328560760875,
            65.28470879768561,
            78.5995605333211,
            13.008782215110587,
            107.68471252503748,
            24.99036806911677
        ])
        .unwrap();
        static ref S4: Series<95> = Series::new(&[
            32.011291185829386,
            41.16885944087332,
            40.942858434710736,
            106.10924460916148,
            75.53203299345262,
            73.95149893049506,
            14.872097219497107,
            67.78751630697572,
            82.77231740415732,
            42.7241543519345,
            77.64589193969962,
            35.85791558401924,
            56.153301785667736,
            105.13900149413487,
            74.35031755858905,
            108.05803668120333,
            51.90499484638232,
            68.87966183273122,
            110.49052926071727,
            75.22123829890211,
            107.77010808492001,
            4.607165988546264,
            56.15610291849866,
            99.14605022210742,
            101.49955820924457,
            72.03998325986862,
            31.30568822373622,
            90.78472112158481,
            103.37900554093656,
            47.919508052764094,
            18.237373695991618,
            13.27391860912538,
            29.159433643868773,
            45.06060362988455,
            0.6141632066634193,
            15.422793069829462,
            100.48797673241812,
            89.3576218159534,
            119.01587846209448,
            71.8670875386896,
            117.37847457893182,
            90.60125715586405,
            78.55976325363845,
            90.10002408578609,
            37.91864519496888,
            6.621009961501429,
            11.690967239164777,
            119.9202419119762,
            83.55433658190775,
            62.95903934829104,
            16.343347086671592,
            51.24043631177927,
            70.39110816261291,
            57.631583291713774,
            10.698063492970508,
            92.41569299379307,
            15.38866720150581,
            83.76735225720627,
            12.514868866764784,
            1.5862487888732266,
            32.87819903597594,
            77.69443311609979,
            7.6287902239824295,
            69.30587949890983,
            83.91005839675323,
            43.751888690665886,
            83.77756135448969,
            86.65631034743015,
            99.68028988552462,
            66.00529313906337,
            52.0814234902964,
            81.78960267875445,
            78.69205444821343,
            119.4361655601349,
            37.81275632823978,
            40.408788061787554,
            119.00525900890923,
            61.66297729091732,
            45.52840580615155,
            66.07547267633618,
            39.27152038672825,
            36.528383513130024,
            15.06016278468826,
            91.25218213697748,
            66.53422079067064,
            1.6410848308236559,
            36.80628261169339,
            13.543602205248789,
            45.247503361751555,
            98.90579766450963,
            80.2899059149801,
            61.47596810076844,
            17.06509748484622,
            77.9244140972354,
            101.50016207442759
        ])
        .unwrap();
    }

    #[test]
    fn test_mean() {
        let s1_mean = S1.mean().map(|m| m.trunc_with_scale(4));
        assert_eq!(s1_mean, Decimal::from_f64(69.6603));
        let s2_mean = S2.mean().map(|m| m.trunc_with_scale(4));
        assert_eq!(s2_mean, Decimal::from_f64(62.8090));
        let s3_mean = S3.mean().map(|m| m.trunc_with_scale(4));
        assert_eq!(s3_mean, Decimal::from_f64(58.4031));
        let s4_mean = S4.mean().map(|m| m.trunc_with_scale(4));
        assert_eq!(s4_mean, Decimal::from_f64(61.1240));
    }

    #[test]
    fn test_median() {
        let s1_median = S1.median().map(|m| m.trunc_with_scale(4));
        assert_eq!(s1_median, Decimal::from_f64(66.6653));
        let s2_median = S2.median().map(|m| m.trunc_with_scale(4));
        assert_eq!(s2_median, Decimal::from_f64(72.7300));
        let s3_median = S3.median().map(|m| m.trunc_with_scale(4));
        assert_eq!(s3_median, Decimal::from_f64(65.2847));
        let s4_median = S4.median().map(|m| m.trunc_with_scale(4));
        assert_eq!(s4_median, Decimal::from_f64(66.0754));
    }

    #[test]
    fn test_sum() {
        let s1_sum = Some(S1.sum().trunc_with_scale(5));
        assert_eq!(s1_sum, Decimal::from_f64(696.60362));
        let s2_sum = Some(S2.sum().trunc_with_scale(5));
        assert_eq!(s2_sum, Decimal::from_f64(1256.18075));
        let s3_sum = Some(S3.sum().trunc_with_scale(5));
        assert_eq!(s3_sum, Decimal::from_f64(2628.14206));
        let s4_sum = Some(S4.sum().trunc_with_scale(5));
        assert_eq!(s4_sum, Decimal::from_f64(5806.78652));
    }

    #[test]
    fn test_geometric_mean() {
        let s1_gm = S1.geometric_mean().map(|gm| gm.trunc_with_scale(5));
        assert_eq!(s1_gm, Decimal::from_f64(62.77311));
        // FIXME: Geometric mean is failing for below series, it panics with Multiplication Overflow
        let s2_gm = S2.geometric_mean().map(|gm| gm.trunc_with_scale(5));
        assert_eq!(s2_gm, Decimal::from_f64(40.71188));
        let s3_gm = S3.geometric_mean().map(|gm| gm.trunc_with_scale(5));
        assert_eq!(s3_gm, Decimal::from_f64(42.16270));
        let s4_gm = S4.geometric_mean().map(|gm| gm.trunc_with_scale(5));
        assert_eq!(s4_gm, Decimal::from_f64(45.55146));
    }

    #[test]
    fn test_max() {
        assert_eq!(S1.max(), Decimal::from_f64(118.99248854518666).as_ref());
        assert_eq!(S2.max(), Decimal::from_f64(123.21554886683052).as_ref());
        assert_eq!(S3.max(), Decimal::from_f64(119.50486249261775).as_ref());
        assert_eq!(S4.max(), Decimal::from_f64(119.9202419119762).as_ref());
    }

    #[test]
    fn test_min() {
        assert_eq!(S1.min(), Decimal::from_f64(27.90710472204815).as_ref());
        assert_eq!(S2.min(), Decimal::from_f64(1.452828443496243).as_ref());
        assert_eq!(S3.min(), Decimal::from_f64(2.068537995180982).as_ref());
        assert_eq!(S4.min(), Decimal::from_f64(0.6141632066634193).as_ref());
    }
}
