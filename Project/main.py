import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
import time
import os
import sys


class DataCleaningAgent:

    def __init__(self):
        self.log = []
        self.report = {}
        self.df_original = None
        self.df_dirty = None
        self.df_clean = None
        self.true_anomaly_mask = None
        self.decisions = []

    def _log(self, message, level="INFO"):
        timestamp = time.strftime('%H:%M:%S')
        entry = f"[{timestamp}][{level}] {message}"
        self.log.append(entry)
        print(entry)

    def _decide(self, reason, decision):
        entry = f"  >> РІШЕННЯ: {decision}  (причина: {reason})"
        self.decisions.append({'reason': reason, 'decision': decision})
        print(entry)

    def _separator(self, title=""):
        line = "=" * 55
        if title:
            print(f"\n{line}\n  {title}\n{line}")
        else:
            print(line)

    def cli_intro(self):
        self._separator("AI-АГЕНТ ОЧИЩЕННЯ ДАНИХ v1.0")
        print("Агент автоматично аналізує датасет, приймає")
        print("рішення про методи очищення і генерує звіт.")
        self._separator()

    def cli_ask_file(self):
        print("\nКрок 1/9 — Завантаження даних")
        default = "AEP_hourly.csv"
        answer = input(f"  Введіть ім'я CSV файлу [{default}]: ").strip()
        if not answer:
            answer = default
        if not os.path.exists(answer):
            print(f"  [ПОМИЛКА] Файл '{answer}' не знайдено.")
            sys.exit(1)
        return answer

    def cli_ask_nrows(self):
        default = 10000
        answer = input(f"  Кількість рядків для аналізу [{default}]: ").strip()
        try:
            return int(answer) if answer else default
        except ValueError:
            return default

    def cli_ask_contamination(self):
        print("\n  Рівень чутливості до аномалій:")
        print("  [1] Низький  — 2%  (мало помилкових спрацювань)")
        print("  [2] Середній — 5%  (рекомендовано)")
        print("  [3] Високий  — 10% (агресивне очищення)")
        answer = input("  Ваш вибір [2]: ").strip()
        mapping = {'1': 0.02, '2': 0.05, '3': 0.10}
        return mapping.get(answer, 0.05)

    def cli_ask_save(self):
        answer = input("\n  Зберегти очищений датасет у CSV? [y/n, default=y]: ").strip().lower()
        return answer != 'n'

    def load_data(self, filepath, nrows):
        self._separator("КРОК 1: СПРИЙНЯТТЯ ДАНИХ")
        self._log(f"Завантаження файлу: {filepath}")
        df = pd.read_csv(filepath, nrows=nrows)
        df.columns = ['Datetime', 'MW']
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        self._log(f"Завантажено рядків: {len(df)}")
        self._log(f"Діапазон: {df['Datetime'].min()} — {df['Datetime'].max()}")
        self._log(f"MW: min={df['MW'].min():.1f}, max={df['MW'].max():.1f}, mean={df['MW'].mean():.1f}")
        self.df_original = df.copy()
        self.report['filepath'] = filepath
        self.report['nrows'] = len(df)
        return df

    def inject_problems(self, df):
        self._separator("КРОК 2: СИМУЛЯЦІЯ ПРОБЛЕМ У ДАНИХ")
        self._log("Введення штучних пропусків та аномалій для демонстрації")
        df = df.copy()
        rng = np.random.default_rng(42)
        n = len(df)

        missing_idx = rng.choice(n, size=int(n * 0.04), replace=False)
        df.loc[missing_idx, 'MW'] = np.nan

        anomaly_idx = rng.choice(n, size=int(n * 0.03), replace=False)
        direction = rng.choice([-1, 1], size=len(anomaly_idx))
        df.loc[anomaly_idx, 'MW'] = (
            self.df_original['MW'].mean() + direction * self.df_original['MW'].std() * 7
        )

        true_mask = np.zeros(n, dtype=bool)
        true_mask[anomaly_idx] = True
        self.true_anomaly_mask = true_mask

        self._log(f"Введено пропусків:  {df['MW'].isna().sum()} ({df['MW'].isna().mean()*100:.1f}%)")
        self._log(f"Введено аномалій:   {len(anomaly_idx)} ({len(anomaly_idx)/n*100:.1f}%)")
        self.df_dirty = df.copy()
        return df

    def diagnose(self, df):
        self._separator("КРОК 3: ДІАГНОСТИКА ТА ПРИЙНЯТТЯ РІШЕНЬ")
        missing_count = df['MW'].isna().sum()
        missing_pct   = missing_count / len(df) * 100
        mean_val = df['MW'].mean()
        std_val  = df['MW'].std()
        skew_val = df['MW'].skew()
        q1 = df['MW'].quantile(0.25)
        q3 = df['MW'].quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = ((df['MW'] < q1 - 1.5*iqr) | (df['MW'] > q3 + 1.5*iqr)).sum()

        self._log(f"Пропусків:          {missing_count} ({missing_pct:.1f}%)")
        self._log(f"Середнє / Std:      {mean_val:.2f} / {std_val:.2f}")
        self._log(f"Асиметрія (skew):   {skew_val:.3f}")
        self._log(f"Аномалій (IQR):     {iqr_outliers}")

        self.report.update({
            'missing_count': int(missing_count),
            'missing_pct':   round(missing_pct, 2),
            'skew_before':   round(skew_val, 3),
            'iqr_outliers':  int(iqr_outliers),
        })

        self._separator("АГЕНТ ПРИЙМАЄ РІШЕННЯ")

        if missing_pct < 2:
            fill_method = 'interpolation'
            self._decide(f"пропусків лише {missing_pct:.1f}% — безпечно інтерполювати", "лінійна інтерполяція")
        elif missing_pct < 8:
            fill_method = 'median'
            self._decide(f"пропусків {missing_pct:.1f}% — медіана надійніша за mean", "заповнення медіаною")
        else:
            fill_method = 'mean'
            self._decide(f"пропусків {missing_pct:.1f}% — забагато для інтерполяції", "заповнення середнім")

        if abs(skew_val) > 1.0:
            scale_after = True
            self._decide(f"асиметрія {skew_val:.2f} > 1.0 — дані мають викиди", "застосувати RobustScaler після очищення")
        else:
            scale_after = False
            self._decide(f"асиметрія {skew_val:.2f} ≤ 1.0 — розподіл прийнятний", "масштабування не потрібне")

        return fill_method, scale_after

    def fill_missing(self, df, method):
        self._separator("КРОК 4: ЗАПОВНЕННЯ ПРОПУСКІВ")
        self._log(f"Обраний метод: {method}")
        df = df.copy()
        before = df['MW'].isna().sum()

        if method == 'interpolation':
            df['MW'] = df['MW'].interpolate(method='linear').ffill().bfill()
        elif method == 'median':
            df['MW'] = df['MW'].fillna(df['MW'].median())
        else:
            df['MW'] = df['MW'].fillna(df['MW'].mean())

        after = df['MW'].isna().sum()
        self._log(f"Пропусків до:   {before}")
        self._log(f"Пропусків після:{after}")
        self.report['missing_after_fill'] = int(after)
        return df

    def detect_anomalies_iqr(self, df):
        self._separator("КРОК 5: ВИЯВЛЕННЯ АНОМАЛІЙ — IQR")
        q1  = df['MW'].quantile(0.25)
        q3  = df['MW'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df['MW'] < lower) | (df['MW'] > upper)
        self._log(f"Межі IQR: [{lower:.2f}, {upper:.2f}]")
        self._log(f"Виявлено аномалій IQR: {mask.sum()}")
        return mask

    def detect_anomalies_if(self, df, contamination):
        self._separator("КРОК 6: ВИЯВЛЕННЯ АНОМАЛІЙ — ISOLATION FOREST")
        self._log(f"Contamination: {contamination}")
        clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        values = df['MW'].values.reshape(-1, 1)
        labels = clf.fit_predict(values)
        mask = (labels == -1)
        self._log(f"Виявлено аномалій IF: {mask.sum()} ({mask.mean()*100:.1f}%)")
        return mask

    def combine_and_fix(self, df, iqr_mask, if_mask):
        self._separator("КРОК 7: КОНСОЛІДАЦІЯ ТА ВИПРАВЛЕННЯ АНОМАЛІЙ")
        combined = iqr_mask | if_mask
        only_iqr = (iqr_mask & ~if_mask).sum()
        only_if  = (~iqr_mask & if_mask).sum()
        both     = (iqr_mask & if_mask).sum()
        self._log(f"Тільки IQR:          {only_iqr}")
        self._log(f"Тільки IF:           {only_if}")
        self._log(f"Обидва методи:       {both}")
        self._log(f"Разом унікальних:    {combined.sum()}")

        self._decide(
            "обидва методи доповнюють один одного",
            f"замінити {combined.sum()} аномальних значень інтерполяцією"
        )

        df = df.copy()
        df['is_anomaly'] = combined
        df.loc[combined, 'MW'] = np.nan
        df['MW'] = df['MW'].interpolate(method='linear').ffill().bfill()

        self.report['anomalies_total'] = int(combined.sum())
        return df, combined

    def apply_robust_scaling(self, df):
        self._separator("КРОК 8: ROBUST МАСШТАБУВАННЯ")
        self._log("Застосування RobustScaler (стійкий до викидів)")
        scaler = RobustScaler()
        df = df.copy()
        df['MW_scaled'] = scaler.fit_transform(df[['MW']]).flatten()
        self._log(f"MW_scaled: mean={df['MW_scaled'].mean():.4f}, std={df['MW_scaled'].std():.4f}")
        self.report['scaling_applied'] = True
        return df

    def compute_metrics(self, combined_mask):
        self._separator("КРОК 9: МЕТРИКИ ЯКОСТІ АГЕНТА")
        true_labels = self.true_anomaly_mask.astype(int)
        pred_labels = combined_mask.values.astype(int)

        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall    = recall_score(true_labels, pred_labels, zero_division=0)
        f1        = f1_score(true_labels, pred_labels, zero_division=0)

        orig_vals  = self.df_original['MW'].values
        clean_vals = self.df_clean['MW'].values
        rmse = np.sqrt(np.mean((orig_vals - clean_vals) ** 2))
        mae  = np.mean(np.abs(orig_vals - clean_vals))

        self._log(f"Precision (точність виявлення):  {precision:.4f}")
        self._log(f"Recall    (повнота виявлення):   {recall:.4f}")
        self._log(f"F1-score:                        {f1:.4f}")
        self._log(f"RMSE (очищені vs оригінал):      {rmse:.4f} МВт")
        self._log(f"MAE  (очищені vs оригінал):      {mae:.4f} МВт")

        self.report.update({
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
            'rmse':      round(rmse, 4),
            'mae':       round(mae, 4),
        })

    def visualize(self, combined_mask):
        self._separator("КРОК 10: ВІЗУАЛІЗАЦІЯ")
        n = min(800, len(self.df_dirty))
        dates = self.df_dirty['Datetime'].iloc[:n]

        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle('AI-агент очищення даних: повний pipeline', fontsize=14, fontweight='bold')

        axes[0].plot(dates, self.df_original['MW'].iloc[:n], color='steelblue', lw=0.8)
        axes[0].set_title('1. Оригінальні чисті дані')
        axes[0].set_ylabel('MW'); axes[0].grid(alpha=0.3)

        axes[1].plot(dates, self.df_dirty['MW'].iloc[:n], color='gray', lw=0.7, alpha=0.7, label='Дані')
        miss = self.df_dirty['MW'].iloc[:n].isna()
        axes[1].scatter(dates[miss], [self.df_dirty['MW'].mean()] * miss.sum(),
                        color='orange', s=12, zorder=5, label=f'Пропуски ({miss.sum()})')
        true_n = self.true_anomaly_mask[:n]
        axes[1].scatter(dates[true_n], self.df_dirty['MW'].iloc[:n][true_n],
                        color='red', s=12, zorder=6, label=f'Аномалії ({true_n.sum()})')
        axes[1].set_title('2. Забруднені дані (пропуски + аномалії)')
        axes[1].set_ylabel('MW'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        detected_n = combined_mask.values[:n]
        axes[2].plot(dates, self.df_dirty['MW'].iloc[:n], color='gray', lw=0.6, alpha=0.6, label='Дані')
        axes[2].scatter(dates[detected_n], self.df_dirty['MW'].fillna(0).iloc[:n][detected_n],
                        color='tomato', s=15, zorder=5, label=f'Виявлено агентом ({detected_n.sum()})')
        axes[2].set_title('3. Виявлені аномалії (IQR + Isolation Forest)')
        axes[2].set_ylabel('MW'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        axes[3].plot(dates, self.df_clean['MW'].iloc[:n], color='seagreen', lw=0.8)
        axes[3].set_title('4. Очищені дані')
        axes[3].set_ylabel('MW'); axes[3].set_xlabel('Час'); axes[3].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('cleaning_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        self._log("Графік збережено: cleaning_result.png")

    def save_result(self, df, filepath):
        out = filepath.replace('.csv', '_cleaned.csv')
        df[['Datetime', 'MW']].to_csv(out, index=False)
        self._log(f"Очищений датасет збережено: {out}")
        self.report['output_file'] = out

    def print_final_report(self):
        self._separator("ПІДСУМКОВИЙ ЗВІТ АГЕНТА")
        print(f"  Файл:                    {self.report.get('filepath')}")
        print(f"  Рядків оброблено:        {self.report.get('nrows')}")
        print(f"  Пропусків виявлено:      {self.report.get('missing_count')} ({self.report.get('missing_pct')}%)")
        print(f"  Пропусків після обробки: {self.report.get('missing_after_fill')}")
        print(f"  Аномалій виявлено:       {self.report.get('anomalies_total')}")
        print(f"  Масштабування:           {'так' if self.report.get('scaling_applied') else 'не потрібне'}")
        print()
        print(f"  Precision:  {self.report.get('precision')}")
        print(f"  Recall:     {self.report.get('recall')}")
        print(f"  F1-score:   {self.report.get('f1')}")
        print(f"  RMSE:       {self.report.get('rmse')} МВт")
        print(f"  MAE:        {self.report.get('mae')} МВт")
        print()
        print("  Прийняті рішення агента:")
        for i, d in enumerate(self.decisions, 1):
            print(f"  {i}. {d['decision']}")
            print(f"     Причина: {d['reason']}")
        if self.report.get('output_file'):
            print(f"\n  Результат збережено: {self.report.get('output_file')}")
        self._separator()

    def run(self):
        self.cli_intro()

        filepath = self.cli_ask_file()
        nrows    = self.cli_ask_nrows()
        contamination = self.cli_ask_contamination()
        save = self.cli_ask_save()

        df = self.load_data(filepath, nrows)
        df = self.inject_problems(df)

        fill_method, scale_after = self.diagnose(df)

        df = self.fill_missing(df, fill_method)
        iqr_mask = self.detect_anomalies_iqr(df)
        if_mask  = self.detect_anomalies_if(df, contamination)
        df, combined_mask = self.combine_and_fix(df, iqr_mask, if_mask)

        if scale_after:
            df = self.apply_robust_scaling(df)

        self.df_clean = df.copy()
        self.compute_metrics(combined_mask)
        self.visualize(combined_mask)

        if save:
            self.save_result(df, filepath)

        self.print_final_report()


if __name__ == '__main__':
    agent = DataCleaningAgent()
    agent.run()
