# H1 CFM pLTV / UA Seed Optimization – Session Summary
1️⃣ Business Context
- Product: CFM (newly launched in Southeast Asia)
- Data collection start date: 2025-12-16
- Objective: Build pLTV (predicted LTV30) model for:  UA seed optimization (select high-quality users to send to ad networks for lookalike expansion)

2️⃣ Identity & Key Mapping
Confirmed Unique Identifier
- vopenid = unique user identifier
- Confirmed by: COUNT(DISTINCT vopenid)
- One-to-one with fuin in etl_recharge

Gameplay tables do NOT use vopenid
- etl_game_detail does NOT contain vopenid.
It contains:
- roleid
- playeropenid
Therefore, Required mapping: `vopenid → roleid`
- Mapping source: etl_new_register has both vopenid and roleid
So all gameplay features must:
1. Aggregate on roleid
2. Join back to vopenid via register table

3️⃣ Tables Used
Core Tables
Table	Purpose
std_master_user_profile	Install & UA attribution
etl_new_register	vopenid ↔ roleid mapping
etl_login	Login behavior
etl_game_detail	Gameplay behavior
etl_recharge	Payment data
4️⃣ Important Column Learnings
etl_login:
- vip column → actually IP-like values
- viplevel → always 0 in current data window
- So do NOT use VIP features

etl_game_detail:
- Has 300+ columns: tdbank_imp_date, worldid, ip, __tablename, gamesvrid, dteventtime, vgameappid, platid, izoneareaid, playerid, level, mapid, gametype, gamemode, dtgamestarttime, gameduration, playerduration, survivalduration, gainedexp, totalexp, gainedgamepoint, totalgamepoint, gainedgold, totalgold, gaineddiamond, totaldiamond, score, rankingame, damagevalue, timeskill, timesbekilled, chapterid, roundid, resurrectiontimes, firstgameflag, unitid, difficulty, gameresult, isboss, dropmatchflag, roomid, dsasvrid, chapterstar, matchmodule, gainedhiddenscore, totalhiddenscore, gainedladderscore, totalladderscore, playeropenid, playercamp, wincamp, totaltimeskill, totaltimesbekilled, totaltimesgame, totaltimeswin, totaltimeslose, totaltimesdraw, isnewbie, hardlevel, timesuserebirthcoin, gainclanactivity, isteam, displayweaponid, ischooserandmap, playergamenick, quitreason, roleid, getacecnt, silveracecnt, awardhonorpoint, gameachievementcnt, gameachievements, headshottimes, beenheadshottimes, throwweaponkill, beenkilledbythrowweapon, meleeweaponkill, gainpropcnt, gainprops, ctkillcnt, tkillcnt, totaltakendamage, zombieinfecthumancnt, killzombiebymelee, teckpioint, topcontinuekill, bulletshotcnt, bulletshothitcnt, killflag, weakpointdamage, winround, roomcreatetime, plantc4times, defusec4times, deadtime, towercnt, towerlvupcnt, towerkillcnt, basehitcnt, canignoregamestat, plantc4cnt, defusec4cnt, hasclanmemberingame, timesthrowweapon, iscrosszone, gspzoneid, breakoutcount, ladderseason, ladderlevel, ladderstage, battleid, roomtype, rankinds, teamnum, teamid, timescore, killlscore, survivalscore, timesassists, clanid, clanname, networkquality, lobbyteamid, avgh1z1score, lastavgh1z1score, rescuescore, h1z1seasonid, knockdowntimes, haveheroweapon, rescuecnt, destoryvehiclecnt, treatval, takevehicledistance, movedistance, maxkilldistance, roompoolid, luckyredbagcnt, nianredbagcnt, survivalmode, rebelcnt, regularcnt, remainrebel, remainregular, open_h1z1_laddermatch, killpolicecnt, killclowncnt, occupybriefcasecnt, cardusedcnt, clownlivetime, clownroundcnt, groupid, viplevel, lobbyteamzoneid, plantpropsnum, recyclepropsnum, destroypropsnum, destroywallnum, evolveexppoint, gamemodetype, creditscore, boardscore, displayweaponvlvl, isinchampion, isboard, realsurvivaltime, killzombiecnt, killzombietypecntarr, isbackflowuser, doublekillcnt, triplekillcnt, quadrakillcnt, pentakillcnt, sniperkillcnt, snipershootcnt, sniperhitcnt, sniperhitheadcnt, throwweaponhitcnt, penetratekillcnt, roundmvpcnt, ladderdownlevelscoredelta, ladderdownlevelscoreused, jumpstarcnt, jumpminpasstime, jumpminlifecost, jumpbestcollectcnt, jumprankinspeed, arenabattleid, airobotcnt, pingmax, pingmin, pingchangecount, pingover300, playerdistance, courageroomgaindelta, couragescore, isairobot, warmabtestindex, isaireplace, campkdinfostr, campopenid
- No vopenid: Must use roleid
Many numeric metrics stored as varchar
→ must use `TRY_CAST()` before aggregation

Important gameplay fields: gameresult, gameduration, score, timeskill, timesbekilled, timesassists, ladderlevel, gamemode, gamemodetype, networkquality, pingmax, pingmin, 

etl_recharge:
Important columns: vopenid, imoney_us, imoney, ds
Confirmed: `COUNT(DISTINCT fuin) == COUNT(DISTINCT vopenid)`
So fuin = vopenid

5️⃣ What "ds" Means
ds = partition date (ingestion date) - the date data is written to the table
It is NOT always equal to event timestamp.

When filtering: `WHERE ds >= DATE '2025-12-16'`
means: 
- Scan partitions from that date
- Efficient for Iceberg
For training windows we use: `p.ds BETWEEN install_date AND install_date + window`

6️⃣ Modeling Design
We designed a D0–D7 feature window and D0–D30 label window.

## Feature & Label Definitions
🎯 Cohort Definition
From: std_master_user_profile
- install_date = `CAST(install_time AS date)`
- Only include installs:
    - ≥ 2025-12-16
- ≤ current_date - 30 (so full label window exists)

🧠 Feature Window
feat_days = 7
Meaning:
Use user behavior from: install_date → install_date + 7
We call these: D7 features

💰 Label Window
label_days = 30
Meaning:
Compute: install_date → install_date + 30
Labels:
- ltv30
- is_payer_30

## Built Feature Categories
1️⃣ UA Attribution Features
From std_master_user_profile:
- media_source
- campaign_id
- adset_id
- ad_id
- site_id
- first_os
- country
- login_channel

These are used later for:
- Seed analysis by campaign
- Model slicing
- Lookalike optimization

2️⃣ Login D7 Features
From etl_login
Examples:
- login_rows_d7
- active_days_d7
- loginchannel_variety_d7
- network_variety_d7
- clientversion_variety_d7
- max_level_seen_d7
- max_ladderscore_d7

3️⃣ Gameplay D7 Features
From etl_game_detail (via roleid)
Examples:
- games_d7
- win_rate_d7
- avg_game_duration_d7
- avg_score_d7
- kills_d7
- deaths_d7
- assists_d7
- kd_d7
- max_level_game_d7
- max_ladderlevel_d7

Note:
We temporarily assumed:
gameresult = 1 → win
Needs domain confirmation.

4️⃣ Payment Features
From etl_recharge
D7:
- rev_d7
- txn_cnt_d7
- first_charge_day_offset_d7

D30:
- ltv30
- is_payer_30
Revenue logic: COALESCE(imoney_us, imoney)

## Final Output Structure
One flat table:
vopenid
roleid
install_date
UA attributes
login D7 features
game D7 features
payment D7 features
ltv30
is_payer_30

Exportable for:
- Python model training
- Offline evaluation
- Seed selection ranking
- Agent-based reporting

## What We Removed / Fixed

❌ Removed VIP features
❌ Removed incorrect vopenid in gameplay
❌ Removed hallucinated join path
❌ Removed repeated evaluation metrics discussion

## Important Observations
- CFM only has data since Dec 16
- No historical behavior before that
- viplevel currently meaningless
- gamereresult distribution must be interpreted carefully
- Many numeric gameplay columns are varchar

## If Starting New Conversation
Use this as bootstrap context:

We are building a pLTV model for CFM (launched 2025-12-16 SEA).
vopenid is the unique user ID.
Gameplay uses roleid; must map via etl_new_register.
Feature window = D0–D7.
Label = LTV30.
Tables: std_master_user_profile, etl_new_register, etl_login, etl_game_detail, etl_recharge.
All gameplay numeric fields require TRY_CAST.
Revenue = COALESCE(imoney_us, imoney).
We want UA seed optimization.

## If Instructing Windsurf to Generate a Report
Tell it:

Sample N users from view
Show feature distribution
Show LTV30 distribution
Show payer rate
Slice by media_source
Show correlation of D7 rev vs LTV30
Show win_rate vs LTV30
Show KD vs LTV30

# H1 Streamlit Application Development

## Overview
Built a comprehensive Streamlit web application for CFM pLTV modeling and UA optimization. The app provides end-to-end workflow from data upload through model training, evaluation, and business simulation.

## Application Structure

### Section A: Key Functions
1. **Data Upload** — CSV upload with flexible 1/2/3-way split, Dataset Registry management
2. **Notebooks** — Embedded analysis notebooks

### Section B: Analysis Reports
**pLTV 30d Analysis** (collapsible section, default: collapsed)
- **Definition** — Business context, Lorenz curve, whale economy
- **Features & Model** — Feature profiling, XGBoost training, model registry
- **Model Evaluation** — Lift curves, Precision@K, ROC/AUC, multi-strategy comparison
- **Action & Simulation** — UA budget allocation simulator
- **Cohort Stability** — Time dynamics, robustness checks, A/B test planning
- **Diagnostics** — Model stability, media source drift

**Standalone Reports**
- **Late Payer Analysis** — Deep-dive into rev_d7=0 segment (ML incremental value)
- **Temporal Analysis** — Cohort-level quality evolution over install dates

### Section C: Data Registry
- Per-page dataset binding (each page remembers its own dataset)
- Sidebar selector with dataset info (rows, size, file)
- Management UI: rename, delete with cascade warnings

## Dataset Registry
- Per-page dataset binding (each page remembers its own dataset)
- Sidebar selector with dataset info (rows, size, file)
- Management UI: rename, delete with cascade warnings
- Auto-migration of legacy cfm_pltv_train/test1/test2 files
- Flexible upload: split into 1, 2, or 3 named datasets

## Analytical Studies (5 Reports in /reports/)
1. **Temporal Analysis** — Install volume, payer rates, ARPU trends by cohort
2. **Cohort Comparison** — ARPU by media source, OS, engagement profiles
3. **Causal Inference** — Behavioral predictors of late conversion
4. **Seed Optimization** — Enriched seeds vs D7-only vs oracle strategies
5. **Real-Time Scoring** — D1/D3/D5/D7 model accuracy comparison
6. **Synthesis Summary** — Cross-study insights and priority actions

# H1 Next Logical Extensions
Mode-level features (gamemode segmentation)
Progression delta (level_day7 - level_day0)
Network quality / ping stability
Fraud heuristics (IP clustering)
Ladder TrueSkill dynamics
First-session depth features