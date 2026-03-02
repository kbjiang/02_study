# Python Data Joins, Aggregations, and Cleaning — Fast Practice Pack (with Solutions)

This is a **decision-ready** pandas practice pack aligned to “standard data joins, aggregations, and cleaning steps.”

- Includes **small synthetic DataFrames (3–5)** you can generate in one cell.
- Includes **24 exercises** with **reference solutions** (pandas-first).
- Datasets are intentionally small so you can run everything quickly.

> Tip: In the interview, narrate: **Clarify → Decompose → Prepare → Compute → Sanity check → Recommend**.

---


## 0) Setup: Create Small Example DataFrames

```python

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# users
n_users = 50
users = pd.DataFrame({
    "user_id": np.arange(1, n_users+1),
    "signup_date": pd.to_datetime("2025-11-01") + pd.to_timedelta(rng.integers(0, 90, size=n_users), unit="D"),
    "plan_type": rng.choice(["free", "pro", "enterprise"], size=n_users, p=[0.6, 0.3, 0.1]),
    "acquisition_channel": rng.choice(["organic", "paid", "referral", "partner"], size=n_users, p=[0.45, 0.35, 0.15, 0.05]),
    "country": rng.choice(["US", "CA", "GB", "IN", "BR"], size=n_users, p=[0.45, 0.1, 0.1, 0.25, 0.1]),
    "device": rng.choice(["ios", "android", "web"], size=n_users, p=[0.3, 0.35, 0.35]),
})

# subscriptions (plan over time)
subs_rows = []
for uid in users.user_id:
    k = rng.choice([0, 1, 2], p=[0.35, 0.5, 0.15])
    start = pd.Timestamp("2025-11-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
    for _ in range(k):
        plan = rng.choice(["pro", "enterprise"], p=[0.8, 0.2])
        dur = int(rng.integers(20, 70))
        end = start + pd.Timedelta(days=dur)
        mrr = 20 if plan == "pro" else 60
        subs_rows.append((uid, start, end, plan, mrr))
        start = end + pd.Timedelta(days=int(rng.integers(0, 15)))  # possible gap
subscriptions = pd.DataFrame(subs_rows, columns=["user_id", "subscription_start", "subscription_end", "plan", "mrr"])
subscriptions["subscription_start"] = pd.to_datetime(subscriptions["subscription_start"])
subscriptions["subscription_end"] = pd.to_datetime(subscriptions["subscription_end"])

# usage (daily)
days = pd.date_range("2026-01-01", periods=60, freq="D")
usage = pd.DataFrame({
    "user_id": rng.choice(users.user_id, size=2500, replace=True),
    "date": rng.choice(days, size=2500, replace=True),
    "sessions": rng.poisson(2, size=2500) + 1,
    "ai_queries": rng.poisson(1, size=2500),
})
launch_date = pd.Timestamp("2026-01-31")
usage.loc[usage["date"] < launch_date, "ai_queries"] = 0

# events (event log)
event_types = ["impression", "click", "signup", "purchase", "video_play", "dashboard_viewed", "login"]
events = pd.DataFrame({
    "user_id": rng.choice(users.user_id, size=6000, replace=True),
    "ts": rng.choice(pd.date_range("2026-01-01", periods=60*24, freq="H"), size=6000, replace=True),
    "event_type": rng.choice(event_types, size=6000, replace=True, p=[0.22, 0.08, 0.05, 0.02, 0.25, 0.18, 0.20]),
    "page": rng.choice(["home", "pricing", "editor", "dashboard", "whats_new"], size=6000, replace=True),
    "video_id": rng.choice(np.arange(1, 80), size=6000, replace=True),
    "watch_minutes": np.maximum(0, rng.normal(6, 3, size=6000)).round(2),
})

# sessions table (optional)
sessions = events.loc[events["event_type"].isin(["login", "dashboard_viewed"])].copy()
sessions = sessions.sort_values(["user_id", "ts"]).reset_index(drop=True)
sessions["session_id"] = sessions.groupby("user_id").cumcount() + 1

users.head(), subscriptions.head(), usage.head(), events.head()


```

---


## A) Joins — Exercises + Solutions

1. **Top-N videos by distinct users (Jan 2026)**  
```python
jan = events[(events.ts >= "2026-01-01") & (events.ts < "2026-02-01")]
top_videos = (jan.groupby("video_id")["user_id"].nunique()
                .sort_values(ascending=False).head(10)
                .reset_index(name="unique_users"))
top_videos
```

2. **CTR for iOS users on “what’s new”**  
```python
ev = events.merge(users[["user_id","device"]], on="user_id", how="left")
ios = ev[ev.device=="ios"]
impr = ios[(ios.event_type=="impression") & (ios.page=="whats_new")].shape[0]
clk  = ios[(ios.event_type=="click") & (ios.page=="whats_new")].shape[0]
ctr = clk / impr if impr else np.nan
{"impressions":impr, "clicks":clk, "ctr":ctr}
```

3. **Left join + COALESCE**  
```python
u_sub = users.merge(subscriptions, on="user_id", how="left")
u_sub["plan"] = u_sub["plan"].fillna("none")
u_sub["mrr"]  = u_sub["mrr"].fillna(0).astype(int)
u_sub[["user_id","plan_type","plan","mrr"]].head(10)
```

4. **Cross join country×device reporting grid**  
```python
grid = (pd.MultiIndex.from_product([users.country.unique(), users.device.unique()],
                                  names=["country","device"]).to_frame(index=False))
counts = users.groupby(["country","device"]).size().reset_index(name="n_users")
report = grid.merge(counts, on=["country","device"], how="left").fillna({"n_users":0})
report.sort_values(["country","device"])
```

5. **Union (concat) logs with different schemas; then join to users**  
```python
events2 = events[["user_id","ts","event_type"]].copy(); events2["source"]="log2"
events1 = events.copy(); events1["source"]="log1"
unioned = pd.concat([events1, events2], ignore_index=True, sort=False)
unioned_u = unioned.merge(users[["user_id","country","device"]], on="user_id", how="left")
unioned_u.head()
```

6. **Funnel by acquisition_channel (impression→click→purchase)**  
```python
ev = events.merge(users[["user_id","acquisition_channel"]], on="user_id", how="left")
f = (ev.pivot_table(index="acquisition_channel", columns="event_type",
                    values="user_id", aggfunc="count", fill_value=0).reset_index())
for c in ["impression","click","purchase"]:
    if c not in f.columns: f[c]=0
f["ctr"] = np.where(f.impression>0, f.click/f.impression, np.nan)
f["purchase_per_click"] = np.where(f.click>0, f.purchase/f.click, np.nan)
f.sort_values("ctr", ascending=False)
```

7. **Range join: attach plan-at-date to usage (daily expansion; OK for small data)**  
```python
subs_daily = []
for r in subscriptions.itertuples(index=False):
    active_days = pd.date_range(r.subscription_start.normalize(), r.subscription_end.normalize(), freq="D")
    subs_daily.append(pd.DataFrame({"user_id":r.user_id, "date":active_days,
                                    "plan_active":r.plan, "mrr_active":r.mrr}))
subs_daily = pd.concat(subs_daily, ignore_index=True) if subs_daily else pd.DataFrame(columns=["user_id","date","plan_active","mrr_active"])

usage_with_plan = usage.merge(subs_daily, on=["user_id","date"], how="left")
usage_with_plan["plan_active"] = usage_with_plan["plan_active"].fillna("none")
usage_with_plan["mrr_active"] = usage_with_plan["mrr_active"].fillna(0).astype(int)
usage_with_plan.head()
```

8. **Join explosion + fix by aggregating before joining**  
```python
exploded = users.merge(subscriptions, on="user_id", how="left").merge(usage, on="user_id", how="left")
exploded.shape

usage_user = (usage.groupby("user_id")
                .agg(total_sessions=("sessions","sum"), total_ai=("ai_queries","sum"))
                .reset_index())
fixed = users.merge(usage_user, on="user_id", how="left").merge(subscriptions[["user_id","plan","mrr"]], on="user_id", how="left")
fixed.shape
```

---


## B) Aggregations — Exercises + Solutions

9. **WoW active users by country/device**  
```python
ev = events.copy()
ev["week"] = ev.ts.dt.to_period("W").dt.start_time
ev_u = ev.merge(users[["user_id","country","device"]], on="user_id", how="left")
wau = (ev_u.groupby(["week","country","device"])["user_id"].nunique()
         .reset_index(name="wau").sort_values(["country","device","week"]))
wau["wau_prev"] = wau.groupby(["country","device"])["wau"].shift(1)
wau["wow_growth"] = np.where(wau.wau_prev>0, (wau.wau-wau.wau_prev)/wau.wau_prev, np.nan)
wau.head(15)
```

10. **Rolling 2-day sessions sum threshold**  
```python
daily = usage.groupby("date")["sessions"].sum().sort_index()
flag = daily.rolling(2).sum()
flag[flag>550].reset_index(name="sessions_2day_sum").head()
```

11. **Weekly funnel rates (impression→click→signup)**  
```python
ev = events.copy()
ev["week"] = ev.ts.dt.to_period("W").dt.start_time
wk = (ev.pivot_table(index="week", columns="event_type", values="user_id",
                     aggfunc="count", fill_value=0).reset_index())
for c in ["impression","click","signup"]:
    if c not in wk.columns: wk[c]=0
wk["ctr"] = np.where(wk.impression>0, wk.click/wk.impression, np.nan)
wk["signup_per_click"] = np.where(wk.click>0, wk.signup/wk.click, np.nan)
wk.head()
```

12. **Cohort retention matrix (signup week, offsets 0..8)**  
```python
u = users.copy()
u["cohort_week"] = u.signup_date.dt.to_period("W").dt.start_time
ev = events.merge(u[["user_id","cohort_week"]], on="user_id", how="left")
ev["event_week"] = ev.ts.dt.to_period("W").dt.start_time
ev["week_offset"] = ((ev.event_week - ev.cohort_week)/np.timedelta64(1,"W")).round().astype(int)
ev = ev[(ev.week_offset>=0) & (ev.week_offset<=8)]
cohort_sizes = u.groupby("cohort_week")["user_id"].nunique()
active = ev.groupby(["cohort_week","week_offset"])["user_id"].nunique().unstack(fill_value=0)
retention = (active.div(cohort_sizes, axis=0)).round(3)
retention.head()
```

13. **MRR by month + new/churned MRR decomposition**  
```python
subs_month = []
for r in subscriptions.itertuples(index=False):
    months = pd.date_range(r.subscription_start.to_period("M").to_timestamp(),
                           r.subscription_end.to_period("M").to_timestamp(), freq="MS")
    subs_month.append(pd.DataFrame({"user_id":r.user_id,"month":months,"plan":r.plan,"mrr":r.mrr}))
subs_month = pd.concat(subs_month, ignore_index=True) if subs_month else pd.DataFrame(columns=["user_id","month","plan","mrr"])

mrr_total = subs_month.groupby("month")["mrr"].sum().sort_index().to_frame("mrr_total")
mrr_total["mrr_prev"] = mrr_total.mrr_total.shift(1)
mrr_total["mrr_delta"] = mrr_total.mrr_total - mrr_total.mrr_prev
mrr_total.head()

new_mrr, churn_mrr = {}, {}
months_sorted = sorted(subs_month["month"].unique())
for m in months_sorted:
    prev = m - pd.offsets.MonthBegin(1)
    now = subs_month[subs_month.month==m].set_index("user_id")["mrr"]
    prv = subs_month[subs_month.month==prev].set_index("user_id")["mrr"] if prev in months_sorted else pd.Series(dtype=float)
    new_mrr[m] = float(now.loc[now.index.difference(prv.index)].sum()) if len(now) else 0.0
    churn_mrr[m] = float(prv.loc[prv.index.difference(now.index)].sum()) if len(prv) else 0.0
pd.DataFrame({"new_mrr":new_mrr, "churned_mrr":churn_mrr}).sort_index().head()
```

14. **Outlier-robust aggregation (winsorize watch_minutes)**  
```python
x = events.watch_minutes
lo, hi = x.quantile([0.01, 0.99])
wins = x.clip(lo, hi)
{"raw_mean":x.mean(), "raw_median":x.median(), "wins_mean":wins.mean(), "p01":lo, "p99":hi}
```

15. **Leadership summary table (last 14 days): adoption + sessions + MRR/user**  
```python
end = usage.date.max(); start = end - pd.Timedelta(days=13)
u14 = usage_with_plan[(usage_with_plan.date>=start) & (usage_with_plan.date<=end)].copy()

u14_user = (u14.groupby(["user_id","plan_active"])
              .agg(total_ai=("ai_queries","sum"), total_sessions=("sessions","sum"))
              .reset_index())
u14_user["adopted_ai"] = u14_user.total_ai > 0

summary = (u14_user.groupby("plan_active")
             .agg(users=("user_id","nunique"),
                  ai_adoption_rate=("adopted_ai","mean"),
                  avg_ai_per_user=("total_ai","mean"),
                  avg_sessions_per_user=("total_sessions","mean"))
             .reset_index())

month = end.to_period("M").to_timestamp()
mrr_users = subs_month[subs_month.month==month].groupby("plan")["user_id"].nunique()
mrr_total = subs_month[subs_month.month==month].groupby("plan")["mrr"].sum()
mrr_per_user = (mrr_total/mrr_users).rename("mrr_per_user").reset_index().rename(columns={"plan":"plan_active"})
summary.merge(mrr_per_user, on="plan_active", how="left")
```

16. **Window-like: time since previous login**  
```python
logins = events[events.event_type=="login"].sort_values(["user_id","ts"]).copy()
logins["prev_ts"] = logins.groupby("user_id")["ts"].shift(1)
logins["hours_since_prev_login"] = (logins.ts - logins.prev_ts)/np.timedelta64(1,"h")
logins.head(10)
```

---


## C) Cleaning & Feature Engineering — Exercises + Solutions

17. **Forward-fill missing sessions per user**  
```python
df = usage.sort_values(["user_id","date"]).copy()
mask = rng.random(len(df)) < 0.02
df.loc[mask, "sessions"] = np.nan
df["sessions_filled"] = df.groupby("user_id")["sessions"].ffill()
df[["user_id","date","sessions","sessions_filled"]].head(20)
```

18. **Explode list column and extract token (“gmail”)**  
```python
apps = pd.DataFrame({
    "user_id": rng.choice(users.user_id, size=200, replace=True),
    "apps": rng.choice([["gmail","chrome"], ["maps"], ["android","gmail"], ["docs","drive"], ["chrome","youtube"]], size=200, replace=True)
})
expl = apps.explode("apps")
gmail_users = expl[expl.apps=="gmail"].user_id.unique()
gmail_users[:10], len(gmail_users)
```

19. **Email regex features: count digits; length before '@'**  
```python
emails = pd.DataFrame({"email": rng.choice(["abc12@gmail.com","user007@company.com","no_digits@yahoo.com","x9x9@foo.org","hello@bar.net"], size=50)})
emails["num_digits"] = emails.email.str.findall(r"\d").str.len()
emails["len_before_at"] = emails.email.str.split("@").str[0].str.len()
emails.head()
```

20. **Sessionization: 30-min inactivity rule**  
```python
ev = events.sort_values(["user_id","ts"]).copy()
ev["prev_ts"] = ev.groupby("user_id")["ts"].shift(1)
ev["gap_min"] = (ev.ts - ev.prev_ts)/np.timedelta64(1,"m")
ev["new_session"] = ev.gap_min.isna() | (ev.gap_min > 30)
ev["session_num"] = ev.groupby("user_id")["new_session"].cumsum()
session_summary = (ev.groupby(["user_id","session_num"])
                     .agg(start=("ts","min"), end=("ts","max"), n_events=("event_type","size"))
                     .reset_index())
session_summary.head()
```

21. **Missingness policy: drop >40% missing; impute others**  
```python
tmp = users.copy()
tmp["maybe_missing"] = rng.choice([None,"x","y"], size=len(tmp), p=[0.5,0.25,0.25])
tmp["mostly_missing"] = rng.choice([None,1], size=len(tmp), p=[0.85,0.15])

missing_pct = tmp.isna().mean().sort_values(ascending=False)
drop_cols = missing_pct[missing_pct>0.40].index.tolist()
tmp2 = tmp.drop(columns=drop_cols)

for c in tmp2.columns:
    if tmp2[c].dtype.kind in "if":
        tmp2[c] = tmp2[c].fillna(tmp2[c].median())
    else:
        tmp2[c] = tmp2[c].fillna("unknown")
tmp2.isna().sum()
```

22. **Deduplicate (user_id, ts, event_type) and quantify impact**  
```python
ev = events.copy()
dups = ev.sample(100, random_state=1)
ev2 = pd.concat([ev, dups], ignore_index=True)

dup_mask = ev2.duplicated(subset=["user_id","ts","event_type"], keep="first")
deduped = ev2.drop_duplicates(subset=["user_id","ts","event_type"], keep="first")
{"rows_before":len(ev2), "rows_after":len(deduped), "duplicates_removed":int(dup_mask.sum())}
```

23. **Data quality checks as code (DQ report)**  
```python
dq = []
dq.append(("subscriptions_mrr_non_negative", int((subscriptions.mrr<0).sum())))
dq.append(("subscriptions_end_before_start", int((subscriptions.subscription_end < subscriptions.subscription_start).sum())))
dq.append(("usage_negative_sessions", int((usage.sessions<0).sum())))
dq.append(("users_user_id_null", int(users.user_id.isna().sum())))
pd.DataFrame(dq, columns=["check","violations"])
```

24. **Column normalization (each column sums to 1; safe for zero-sum)**  
```python
mat = pd.DataFrame(rng.integers(0,10,size=(5,4)), columns=list("ABCD"))
col_sums = mat.sum(axis=0)
norm = mat.div(col_sums.where(col_sums!=0, np.nan), axis=1).fillna(0)
mat, col_sums, norm, norm.sum(axis=0)
```

---


## Interview-mode routine (use for any exercise)

1. **Clarify:** grain, metric definition, time window  
2. **Prepare:** join keys, handle missing/duplicates, choose correct grain  
3. **Compute:** groupby/agg and segmentation  
4. **Sanity check:** invariants, back-of-envelope  
5. **Decision:** 2 bullets on what you’d recommend next
