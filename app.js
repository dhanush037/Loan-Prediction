// Synced from loan_prediction.py's print_js_sync_constants().
// Feature order: [Applicant_Income, Credit_Score, Loan_Amount, Debt_to_Income, Is_Self-Employed, Is_Unemployed]
const SCALER_MEAN = [85818.429167, 559.177083, 251479.395833, 4.254690, 0.308333, 0.370833];
const SCALER_SCALE = [42240.976044, 155.899837, 140809.141919, 4.277155, 0.461805, 0.483028];
const COEF = [1.608054, 3.166321, -1.553622, 0.312045, -0.293842, -2.945177];
const INTERCEPT = -1.187694;

const CREDIT_SCORE_STORAGE_KEY = 'loansense_credit_score';

function restoreSavedCreditScore() {
  try {
    const saved = localStorage.getItem(CREDIT_SCORE_STORAGE_KEY);
    if (saved !== null) {
      document.getElementById('input-credit').value = saved;
    }
  } catch (e) {
    // localStorage unavailable (e.g. private browsing) - fall back to the default value.
  }
}

function persistCreditScore() {
  try {
    const value = document.getElementById('input-credit').value;
    if (value !== '') {
      localStorage.setItem(CREDIT_SCORE_STORAGE_KEY, value);
    }
  } catch (e) {
    // Ignore - persistence is a convenience, not a requirement.
  }
}

restoreSavedCreditScore();
document.getElementById('input-credit').addEventListener('change', persistCreditScore);

// Nav active-state: a link is "active" when the section its href points to
// is the one currently in view. Several links can point at the same
// section (Workflow/Dataset both point at #prediction-engine, Model/Results
// both point at #about), so more than one link may be active at once.
const NAV_LINKS = Array.from(document.querySelectorAll('#main-nav a'));

function setActiveNavSection(sectionId) {
  NAV_LINKS.forEach((link) => {
    const isActive = link.getAttribute('href') === '#' + sectionId;
    link.classList.toggle('text-primary', isActive);
    link.classList.toggle('font-body-bold', isActive);
    link.classList.toggle('text-body-base', !isActive);
    link.classList.toggle('text-on-surface-variant', !isActive);
  });
}

const observedSections = ['home', 'prediction-engine', 'about']
  .map((id) => document.getElementById(id))
  .filter(Boolean);

if (observedSections.length && 'IntersectionObserver' in window) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) {
        setActiveNavSection(visible.target.id);
      }
    },
    { rootMargin: '-80px 0px -60% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] }
  );
  observedSections.forEach((section) => sectionObserver.observe(section));
}

// Currency conversion: the model was trained on USD-scale figures, so any
// non-USD input gets converted to USD before scoring. Rates are USD -> X
// (e.g. currencyRates.INR = how many INR per 1 USD).
const FALLBACK_RATES = {
  USD: 1,
  EUR: 0.92,
  GBP: 0.79,
  INR: 83.5,
  JPY: 149.5,
  CAD: 1.36,
  AUD: 1.52,
};
let currencyRates = { ...FALLBACK_RATES };

async function loadCurrencyRates() {
  try {
    const symbols = Object.keys(FALLBACK_RATES).filter((c) => c !== 'USD').join(',');
    const response = await fetch(`https://api.frankfurter.app/latest?base=USD&symbols=${symbols}`);
    if (!response.ok) throw new Error('Rate fetch failed');
    const data = await response.json();
    currencyRates = { USD: 1, ...data.rates };
  } catch (e) {
    // Network error or API down - the hardcoded FALLBACK_RATES already in
    // currencyRates keep the form usable, just with slightly stale rates.
  }
}

function convertToUSD(amount, currencyCode) {
  const rate = currencyRates[currencyCode] || 1;
  return amount / rate;
}

function updateCurrencySymbols() {
  const select = document.getElementById('input-currency');
  const symbol = select.options[select.selectedIndex].dataset.symbol || '$';
  document.getElementById('income-currency-symbol').innerText = symbol;
  document.getElementById('loan-currency-symbol').innerText = symbol;
}

loadCurrencyRates();
document.getElementById('input-currency').addEventListener('change', updateCurrencySymbols);
updateCurrencySymbols();

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictApproval(income, credit, loan, employment) {
  const selfEmployed = employment === 'self' ? 1 : 0;
  const unemployed = employment === 'unemployed' ? 1 : 0;
  const debtToIncome = loan / income;
  const raw = [income, credit, loan, debtToIncome, selfEmployed, unemployed];

  const scaled = raw.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);

  let z = INTERCEPT;
  const contributions = [];
  for (let i = 0; i < scaled.length; i++) {
    const contribution = scaled[i] * COEF[i];
    z += contribution;
    contributions.push(contribution);
  }

  const probApproved = sigmoid(z);
  return {
    probApproved,
    incomeContribution: contributions[0],
    creditContribution: contributions[1],
    loanContribution: contributions[2],
    dtiContribution: contributions[3],
  };
}

function impactLabel(absContribution) {
  if (absContribution > 1.0) return 'High Impact';
  if (absContribution > 0.4) return 'Mod Impact';
  return 'Low Impact';
}

function barWidth(absContribution) {
  return Math.min(95, Math.max(10, absContribution * 40)) + '%';
}

document.getElementById('prediction-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const currency = document.getElementById('input-currency').value;
  const incomeRaw = parseFloat(document.getElementById('input-income').value);
  const credit = parseFloat(document.getElementById('input-credit').value);
  const loanRaw = parseFloat(document.getElementById('input-loan').value);
  const employment = document.getElementById('input-employment').value;

  const income = convertToUSD(incomeRaw, currency);
  const loan = convertToUSD(loanRaw, currency);

  const result = predictApproval(income, credit, loan, employment);
  const isApproved = result.probApproved > 0.5;
  const confidencePct = (isApproved ? result.probApproved : 1 - result.probApproved) * 100;

  document.getElementById('result-empty').style.opacity = '0';
  setTimeout(() => {
    document.getElementById('result-empty').style.display = 'none';
    document.getElementById('result-content').style.opacity = '1';

    document.getElementById('sim-id').innerText = 'APL-' + Math.floor(Math.random() * 90000 + 10000);

    const badge = document.getElementById('decision-badge');
    const riskLevel = document.getElementById('risk-level');
    const riskBar = document.getElementById('risk-bar');
    const reason = document.getElementById('decision-reason');
    const conf = document.getElementById('conf-score');

    conf.innerText = confidencePct.toFixed(1);

    if (isApproved) {
      badge.innerText = 'Approved';
      badge.className = 'px-6 py-2 rounded-full font-body-bold text-[20px] bg-success-green/10 text-secondary border border-success-green/20 mb-2 transition-all';

      riskLevel.innerText = 'Low Risk';
      riskLevel.className = 'font-body-bold text-secondary';
      riskBar.className = 'bg-secondary h-2 rounded-full transition-all duration-1000';
      riskBar.style.width = (100 - confidencePct).toFixed(0) + '%';
      reason.innerText = "The model's learned coefficients favor approval for this applicant profile.";
    } else {
      badge.innerText = 'Rejected';
      badge.className = 'px-6 py-2 rounded-full font-body-bold text-[20px] bg-error-container text-error border border-error/20 mb-2 transition-all';

      riskLevel.innerText = 'High Risk';
      riskLevel.className = 'font-body-bold text-error';
      riskBar.className = 'bg-error h-2 rounded-full transition-all duration-1000';
      riskBar.style.width = confidencePct.toFixed(0) + '%';
      reason.innerText = "The model's learned coefficients weigh against approval for this applicant profile.";
    }

    const creditAbs = Math.abs(result.creditContribution);
    const incomeAbs = Math.abs(result.incomeContribution);
    const loanAbs = Math.abs(result.loanContribution);
    const dtiAbs = Math.abs(result.dtiContribution);

    document.getElementById('driver-credit-label').innerText = impactLabel(creditAbs);
    document.getElementById('driver-credit-bar').style.width = barWidth(creditAbs);
    document.getElementById('driver-income-label').innerText = impactLabel(incomeAbs);
    document.getElementById('driver-income-bar').style.width = barWidth(incomeAbs);
    document.getElementById('driver-loan-label').innerText = impactLabel(loanAbs);
    document.getElementById('driver-loan-bar').style.width = barWidth(loanAbs);
    document.getElementById('driver-dti-label').innerText = impactLabel(dtiAbs);
    document.getElementById('driver-dti-bar').style.width = barWidth(dtiAbs);
  }, 300);
});
