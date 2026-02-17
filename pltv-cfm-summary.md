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

# H1 Next Logical Extensions
Mode-level features (gamemode segmentation)
Progression delta (level_day7 - level_day0)
Network quality / ping stability
Fraud heuristics (IP clustering)
Ladder TrueSkill dynamics
First-session depth features

## SQL that generates 350MB of data for model training:
```sql
with params AS (
  SELECT
    DATE '2025-12-16' AS data_start,
    7  AS feat_days,
    30 AS label_days
),
/* 1) UA cohort: one row per vopenid with install_date + UA fields */
ua_cohort AS (
  SELECT
    vopenid,
    CAST(install_time AS date) AS install_date,
    game_id,
    media_source,
    campaign_id,
    adset_id,
    ad_id,
    site_id,
    first_os,
    last_os,
    first_country_code,
    last_country_code,
    first_login_channel,
    last_login_channel
  FROM iceberg.cfm_vn.std_master_user_profile
  WHERE vopenid IS NOT NULL
    AND CAST(install_time AS date) >= (SELECT data_start FROM params)
    -- ensure cohort old enough to have label window fully observed
    AND CAST(install_time AS date) <= date_add('day', -(SELECT label_days FROM params), current_date)
),
/* 2) map vopenid -> roleid (choose earliest observed roleid after launch) */
role_map AS (
  SELECT
    vopenid,
    min_by(roleid, ds) AS roleid
  FROM iceberg.cfm_vn.etl_new_register
  WHERE ds >= (SELECT data_start FROM params)
    AND vopenid IS NOT NULL
    AND roleid IS NOT NULL
  GROUP BY 1
),
base AS (
  SELECT
    u.*,
    rm.roleid
  FROM ua_cohort u
  LEFT JOIN role_map rm
    ON u.vopenid = rm.vopenid
),
/* 3) Login features D0-D7 */
login_d7 AS (
  SELECT
    b.vopenid,
    b.install_date,

    COUNT(*) AS login_rows_d7,

    -- exact distinct days (bounded 0..~8, important feature)
    COUNT(DISTINCT CAST(l.dteventtime AS date)) AS active_days_d7,

    -- keep approx for high-volume distincts (swap to COUNT(DISTINCT ...) later if cheap enough)
    approx_distinct(NULLIF(l.loginchannel, '')) AS loginchannel_variety_d7,
    approx_distinct(NULLIF(l.network, ''))       AS network_variety_d7,
    approx_distinct(NULLIF(l.clientversion, '')) AS clientversion_variety_d7,

    MAX(TRY_CAST(l.level AS integer))       AS max_level_seen_d7,
    MAX(TRY_CAST(l.viplevel AS integer))    AS max_viplevel_seen_d7, -- likely 0, harmless
    MAX(TRY_CAST(l.ladderscore AS double))  AS max_ladderscore_d7

  FROM base b
  JOIN iceberg.cfm_vn.etl_login l
    ON l.vopenid = b.vopenid

   -- partition pruning (fast scan)
   AND l.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)

   -- correctness filter (event-time window)
   AND CAST(l.dteventtime AS date)
       BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)

  GROUP BY 1,2
),
/* 4) Gameplay features D0-D7 (via roleid) */
game_d7 AS (
  SELECT
    b.vopenid,
    b.install_date,

    COUNT(*) AS games_d7,

    -- win rate: gameresult distribution varies, treat "1" as win pending domain confirmation
    AVG(CASE WHEN TRY_CAST(g.gameresult AS integer) = 1 THEN 1.0 ELSE 0.0 END) AS win_rate_d7,
    AVG(TRY_CAST(g.gameduration AS double)) AS avg_game_duration_d7,
    AVG(TRY_CAST(g.score AS double))        AS avg_score_d7,
    SUM(COALESCE(TRY_CAST(g.timeskill AS double), 0))     AS kills_d7,
    SUM(COALESCE(TRY_CAST(g.timesbekilled AS double), 0)) AS deaths_d7,
    SUM(COALESCE(TRY_CAST(g.timesassists AS double), 0))  AS assists_d7,
    (SUM(COALESCE(TRY_CAST(g.timeskill AS double), 0)) * 1.0) /
      NULLIF(SUM(COALESCE(TRY_CAST(g.timesbekilled AS double), 0)), 0) AS kd_d7,
    MAX(TRY_CAST(g.level AS integer))       AS max_level_game_d7,
    MAX(TRY_CAST(g.ladderlevel AS double))  AS max_ladderlevel_d7
  FROM base b
  JOIN iceberg.cfm_vn.etl_game_detail g
    ON g.roleid = b.roleid
   AND g.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)
  GROUP BY 1,2
),
/* 5) Payment label + early pay features */
pay_agg AS (
  SELECT
    b.vopenid,
    b.install_date,
    -- D0-D7 pay features
    SUM(
      CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)
           THEN COALESCE(p.imoney_us, p.imoney, 0)
           ELSE 0 END
    ) AS rev_d7,
    SUM(
      CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)
           THEN 1 ELSE 0 END
    ) AS txn_cnt_d7,
    MIN(
      CASE WHEN COALESCE(p.imoney_us, p.imoney, 0) > 0
             AND p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params), b.install_date)
           THEN date_diff('day', b.install_date, p.ds)
           ELSE NULL END
    ) AS first_charge_day_offset_d7,
    -- Label: LTV30
    SUM(
      CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params), b.install_date)
           THEN COALESCE(p.imoney_us, p.imoney, 0)
           ELSE 0 END
    ) AS ltv30,
    CASE WHEN SUM(
      CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params), b.install_date)
           THEN COALESCE(p.imoney_us, p.imoney, 0)
           ELSE 0 END
    ) > 0 THEN 1 ELSE 0 END AS is_payer_30
  FROM base b
  LEFT JOIN iceberg.cfm_vn.etl_recharge p
    ON p.vopenid = b.vopenid
   AND p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params), b.install_date)
  GROUP BY 1,2
)
SELECT
  b.vopenid,
  b.roleid,
  b.install_date,
  -- UA attribution
  b.game_id,
  b.media_source,
  b.campaign_id,
  b.adset_id,
  b.ad_id,
  b.site_id,
  b.first_os,
  b.last_os,
  b.first_country_code,
  b.last_country_code,
  b.first_login_channel,
  b.last_login_channel,
  -- login features
  COALESCE(l.login_rows_d7, 0)            AS login_rows_d7,
  COALESCE(l.active_days_d7, 0)           AS active_days_d7,
  COALESCE(l.loginchannel_variety_d7, 0)  AS loginchannel_variety_d7,
  COALESCE(l.network_variety_d7, 0)       AS network_variety_d7,
  COALESCE(l.clientversion_variety_d7, 0) AS clientversion_variety_d7,
  COALESCE(l.max_level_seen_d7, 0)        AS max_level_seen_d7,
  COALESCE(l.max_ladderscore_d7, 0)       AS max_ladderscore_d7,
  -- gameplay features
  COALESCE(g.games_d7, 0)             AS games_d7,
  COALESCE(g.win_rate_d7, 0)          AS win_rate_d7,
  COALESCE(g.avg_game_duration_d7, 0) AS avg_game_duration_d7,
  COALESCE(g.avg_score_d7, 0)         AS avg_score_d7,
  COALESCE(g.kills_d7, 0)             AS kills_d7,
  COALESCE(g.deaths_d7, 0)            AS deaths_d7,
  COALESCE(g.assists_d7, 0)           AS assists_d7,
  COALESCE(g.kd_d7, 0)                AS kd_d7,
  COALESCE(g.max_level_game_d7, 0)    AS max_level_game_d7,
  COALESCE(g.max_ladderlevel_d7, 0)   AS max_ladderlevel_d7,
  -- pay features + labels
  COALESCE(p.rev_d7, 0)       AS rev_d7,
  COALESCE(p.txn_cnt_d7, 0)   AS txn_cnt_d7,
  p.first_charge_day_offset_d7,
  COALESCE(p.ltv30, 0)        AS ltv30,
  COALESCE(p.is_payer_30, 0)  AS is_payer_30
FROM base b
LEFT JOIN login_d7 l
  ON b.vopenid = l.vopenid AND b.install_date = l.install_date
LEFT JOIN game_d7 g
  ON b.vopenid = g.vopenid AND b.install_date = g.install_date
LEFT JOIN pay_agg p
  ON b.vopenid = p.vopenid AND b.install_date = p.install_date
;```