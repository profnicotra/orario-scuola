# Prototype: hybrid school timetable generator (backtracking), with CSV/ICS export.
# Notes:
# - No external libraries beyond standard Python.
# - Generates a small demo dataset (2 classi, 4 materie, 4 docenti, 1 settimana).
# - Hybrid blocks: lab subjects use macro blocks (2h). If infeasible, optional fallback to split into 1h units.
# - Exports schedule.csv + ICS files per classe e per docente.
#
# Esegui con: python hybrid_timetable_prototype.py

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from datetime import date, timedelta, datetime
import csv
import os

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Class:
    class_id: str
    name: str

@dataclass(frozen=True)
class Subject:
    subject_id: str
    name: str
    block_len_min: int = 1          # minimo blocco richiesto
    macro_preferred: bool = False   # se True, genera macro-lezioni (k=block_len_min)

@dataclass(frozen=True)
class Teacher:
    teacher_id: str
    name: str

@dataclass
class MergeGroup:
    merge_id: str
    name: str
    members: List[str]  # class_id list

@dataclass
class ClassSubjectPlan:
    plan_id: str
    class_or_group_id: str  # "C:<class_id>" or "G:<merge_id>"
    subject_id: str
    total_hours: int

@dataclass
class Availability:
    # Disponibilità per docente per giorno della settimana (0=Mon..6=Sun) e per slot label ("08:00",...)
    teacher_id: str
    weekday: int
    slot_labels: Set[str]

@dataclass
class SchoolCalendar:
    school_days: List[date]

@dataclass
class Lesson:
    lesson_id: str
    class_or_group_id: str    # "C:<class_id>" o "G:<merge_id>"
    subject_id: str
    duration_slots: int       # 1 per unità; >=2 per macro
    split_fallback: bool      # se True e la macro fallisce, spezza in unità da 1h

@dataclass
class Assignment:
    lesson_id: str
    date: date
    start_slot: str
    end_slot: str
    teacher_id: str
    class_or_group_id: str
    subject_id: str


# -----------------------------
# Utilities
# -----------------------------

SLOTS_ALL = ["08:00","09:00","10:00","11:00","12:00","13:00","14:30","15:30"]

def slots_between(start_label: str, k: int, slots: List[str]) -> Optional[List[str]]:
    """Ritorna la lista di k slot consecutivi a partire da start_label, altrimenti None."""
    if start_label not in slots:
        return None
    i = slots.index(start_label)
    j = i + k
    if j <= len(slots):
        block = slots[i:j]
        # Evita attraversare 13:00 -> 14:30 come contiguità
        for a, b in zip(block[:-1], block[1:]):
            if a == "13:00" and b == "14:30":
                return None
        return block
    return None

def weekday_it(d: date) -> str:
    giorni = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
    return giorni[d.weekday()]

def dt_local(date_obj: date, slot_label: str) -> datetime:
    hh, mm = map(int, slot_label.split(":"))
    return datetime(date_obj.year, date_obj.month, date_obj.day, hh, mm)


# -----------------------------
# Solver (backtracking)
# -----------------------------

class TimetableSolver:
    def __init__(
        self,
        classes: Dict[str, Class],
        subjects: Dict[str, Subject],
        teachers: Dict[str, Teacher],
        teacher_subjects: Dict[str, Set[str]],  # teacher_id -> set(subject_id)
        availability: List[Availability],
        class_plans: List[ClassSubjectPlan],
        merge_groups: Dict[str, MergeGroup],
        school_calendar: SchoolCalendar,
        use_afternoon: bool = False,
    ):
        self.classes = classes
        self.subjects = subjects
        self.teachers = teachers
        self.teacher_subjects = teacher_subjects
        self.availability = availability
        self.class_plans = class_plans
        self.merge_groups = merge_groups
        self.school_calendar = school_calendar
        self.use_afternoon = use_afternoon

        self.slots = [s for s in SLOTS_ALL if (use_afternoon or s <= "13:00")]

        # Build availability map: (teacher, weekday) -> set(slot_labels)
        self.avail_map: Dict[Tuple[str,int], Set[str]] = {}
        for av in availability:
            self.avail_map.setdefault((av.teacher_id, av.weekday), set()).update(av.slot_labels)

        # Resource occupancy trackers during search
        self.teacher_busy: Set[Tuple[str, date, str]] = set()
        self.class_busy: Set[Tuple[str, date, str]] = set()
        self.group_busy: Set[Tuple[str, date, str]] = set()

        # Precompute map from group key to its member classes
        # Keys per class_or_group_id: "C:<class_id>" or "G:<merge_id>"
        self.group_members: Dict[str, List[str]] = {}
        for gid, g in merge_groups.items():
            self.group_members[f"G:{gid}"] = list(g.members)

        # Diagnostics
        self.initial_zero_domain: List[str] = []

    def generate_lessons(self, allow_macro_split_fallback: bool = True) -> List[Lesson]:
        lessons: List[Lesson] = []
        counter = 1
        for plan in self.class_plans:
            subj = self.subjects[plan.subject_id]
            k = max(1, subj.block_len_min)
            if subj.macro_preferred and k > 1:
                n_full = plan.total_hours // k
                rem = plan.total_hours % k
                for _ in range(n_full):
                    lessons.append(Lesson(
                        lesson_id=f"L{counter}",
                        class_or_group_id=plan.class_or_group_id,
                        subject_id=plan.subject_id,
                        duration_slots=k,
                        split_fallback=allow_macro_split_fallback
                    ))
                    counter += 1
                if rem > 0:
                    for _ in range(rem):
                        lessons.append(Lesson(
                            lesson_id=f"L{counter}",
                            class_or_group_id=plan.class_or_group_id,
                            subject_id=plan.subject_id,
                            duration_slots=1,
                            split_fallback=False
                        ))
                        counter += 1
            else:
                for _ in range(plan.total_hours):
                    lessons.append(Lesson(
                        lesson_id=f"L{counter}",
                        class_or_group_id=plan.class_or_group_id,
                        subject_id=plan.subject_id,
                        duration_slots=1,
                        split_fallback=False
                    ))
                    counter += 1
        return lessons

    def candidate_assignments(self, lesson: Lesson) -> List[Tuple[date, str, str]]:
        """Restituisce [(data, start_slot, teacher_id)] possibili ignorando conflitti con altre lezioni."""
        candidates = []
        subj = self.subjects[lesson.subject_id]
        k = lesson.duration_slots
        # Docenti abilitati
        teachers_for_subj = [tid for tid, subs in self.teacher_subjects.items() if subj.subject_id in subs]
        if not teachers_for_subj:
            return candidates
        for d in self.school_calendar.school_days:
            wd = d.weekday()
            for start in self.slots:
                block = slots_between(start, k, self.slots)
                if not block:
                    continue
                for tid in teachers_for_subj:
                    av_slots = self.avail_map.get((tid, wd), set())
                    if all((s in av_slots) for s in block):
                        candidates.append((d, start, tid))
        return candidates

    def order_lessons(self, lessons: List[Lesson], domains: Dict[str, List[Tuple[date,str,str]]]) -> List[Lesson]:
        # Euristica: blocchi più lunghi prima; poi dominio più piccolo (MRV); gruppi prima delle classi singole
        def key(l: Lesson):
            dom = domains.get(l.lesson_id, [])
            is_group = l.class_or_group_id.startswith("G:")
            return (-l.duration_slots, len(dom), 0 if is_group else 1, l.lesson_id)
        return sorted(lessons, key=key)

    def is_conflict(self, lesson: Lesson, d: date, start: str, tid: str) -> bool:
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        if not block:
            return True
        # Docente
        for s in block:
            if (tid, d, s) in self.teacher_busy:
                return True
        # Classe/Gruppo
        if lesson.class_or_group_id.startswith("G:"):
            if any((lesson.class_or_group_id, d, s) in self.group_busy for s in block):
                return True
            for cid in self.group_members.get(lesson.class_or_group_id, []):
                if any((cid, d, s) in self.class_busy for s in block):
                    return True
        else:
            cid = lesson.class_or_group_id.split("C:",1)[1]
            if any((cid, d, s) in self.class_busy for s in block):
                return True
        return False

    def place(self, lesson: Lesson, d: date, start: str, tid: str):
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        assert block
        for s in block:
            self.teacher_busy.add((tid, d, s))
            if lesson.class_or_group_id.startswith("G:"):
                self.group_busy.add((lesson.class_or_group_id, d, s))
                for cid in self.group_members.get(lesson.class_or_group_id, []):
                    self.class_busy.add((cid, d, s))
            else:
                cid = lesson.class_or_group_id.split("C:",1)[1]
                self.class_busy.add((cid, d, s))

    def unplace(self, lesson: Lesson, d: date, start: str, tid: str):
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        assert block
        for s in block:
            self.teacher_busy.discard((tid, d, s))
            if lesson.class_or_group_id.startswith("G:"):
                self.group_busy.discard((lesson.class_or_group_id, d, s))
                for cid in self.group_members.get(lesson.class_or_group_id, []):
                    self.class_busy.discard((cid, d, s))
            else:
                cid = lesson.class_or_group_id.split("C:",1)[1]
                self.class_busy.discard((cid, d, s))

    def search(self, lessons: List[Lesson], domains: Dict[str, List[Tuple[date,str,str]]]) -> Optional[List[Assignment]]:
        if not lessons:
            return []

        L = lessons[0]

        def candidate_score(c):
            d, start, tid = c
            base = {"08:00": 5, "09:00": 3, "10:00": 1, "11:00": 1, "12:00": 2, "13:00": 3, "14:30": 6, "15:30": 6}.get(start, 4)
            teacher_load = sum(1 for (tt, dd, ss) in self.teacher_busy if tt == tid and dd == d)
            return (base, teacher_load)  # più basso è meglio

        for (d, start, tid) in sorted(domains[L.lesson_id], key=candidate_score):
            if not self.is_conflict(L, d, start, tid):
                self.place(L, d, start, tid)
                tail = lessons[1:]
                res = self.search(tail, domains)
                if res is not None:
                    block = slots_between(start, L.duration_slots, self.slots)
                    end_slot = block[-1]
                    return [Assignment(L.lesson_id, d, start, end_slot, tid, L.class_or_group_id, L.subject_id)] + res
                self.unplace(L, d, start, tid)
        return None

    def solve(self, allow_macro_split_fallback: bool = True, try_split_on_fail: bool = True):
        # 1) Build lessons (macro + units)
        lessons = self.generate_lessons(allow_macro_split_fallback=allow_macro_split_fallback)

        # 2) Domains
        domains: Dict[str, List[Tuple[date,str,str]]] = {}
        zero_domain_lessons = []
        for L in lessons:
            dom = self.candidate_assignments(L)
            domains[L.lesson_id] = dom
            if not dom:
                zero_domain_lessons.append(L)

        # 3) Split fallback per le macro senza candidati
        if zero_domain_lessons and try_split_on_fail:
            changed = False
            new_lessons = []
            for L in lessons:
                if L.duration_slots > 1 and L.split_fallback and L in zero_domain_lessons:
                    for i in range(L.duration_slots):
                        new_lessons.append(Lesson(
                            lesson_id=f"{L.lesson_id}_u{i+1}",
                            class_or_group_id=L.class_or_group_id,
                            subject_id=L.subject_id,
                            duration_slots=1,
                            split_fallback=False
                        ))
                    changed = True
                else:
                    new_lessons.append(L)
            if changed:
                lessons = new_lessons
                domains = {}
                zero_domain_lessons = []
                for L in lessons:
                    dom = self.candidate_assignments(L)
                    domains[L.lesson_id] = dom
                    if not dom:
                        zero_domain_lessons.append(L)

        self.initial_zero_domain = [L.lesson_id for L in zero_domain_lessons]
        if zero_domain_lessons:
            return None, {"feasible": False, "reason": "Lezioni senza candidati", "lessons": zero_domain_lessons}

        # 4) Order + Search
        ordered_lessons = self.order_lessons(lessons, domains)
        solution = self.search(ordered_lessons, domains)
        if solution is None:
            return None, {"feasible": False, "reason": "Backtracking fallito (probabile conflitto di risorse).", "zero_domain": self.initial_zero_domain}
        solution.sort(key=lambda a: (a.date, a.start_slot, a.class_or_group_id))
        return solution, {"feasible": True, "zero_domain": self.initial_zero_domain}


# -----------------------------
# Export: CSV and ICS
# -----------------------------

def export_csv(schedule: List[Assignment], classes: Dict[str, Class], subjects: Dict[str, Subject], teachers: Dict[str, Teacher], merge_groups: Dict[str, MergeGroup], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Data","Giorno","Inizio","Fine","Classe/Group","Materia","Docente"])
        for a in schedule:
            name = ""
            if a.class_or_group_id.startswith("C:"):
                cid = a.class_or_group_id.split("C:",1)[1]
                name = classes[cid].name
            else:
                gid = a.class_or_group_id.split("G:",1)[1]
                name = merge_groups[gid].name
            w.writerow([a.date.isoformat(), weekday_it(a.date), a.start_slot, a.end_slot, name, subjects[a.subject_id].name, teachers[a.teacher_id].name])

def ics_escape(text: str) -> str:
    return text.replace("\\","\\\\").replace("\n","\\n").replace(",","\\,").replace(";","\\;")

def export_ics_per_class(schedule: List[Assignment], classes: Dict[str, Class], subjects: Dict[str, Subject], teachers: Dict[str, Teacher], merge_groups: Dict[str, MergeGroup], folder: str, tzid: str = "Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    class_events: Dict[str, List[Assignment]] = {cid: [] for cid in classes.keys()}
    for a in schedule:
        if a.class_or_group_id.startswith("C:"):
            cid = a.class_or_group_id.split("C:",1)[1]
            class_events[cid].append(a)
        else:
            gid = a.class_or_group_id.split("G:",1)[1]
            members = merge_groups[gid].members
            for cid in members:
                class_events[cid].append(a)

    for cid, events in class_events.items():
        path = os.path.join(folder, f"orario_classe_{cid}.ics")
        with open(path, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx, a in enumerate(events, start=1):
                start_dt = dt_local(a.date, a.start_slot)
                all_slots = [s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i = all_slots.index(a.end_slot)
                    if i+1 < len(all_slots):
                        end_label = all_slots[i+1]
                        end_dt = dt_local(a.date, end_label)
                    else:
                        end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)
                else:
                    end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)

                def fmt(dt: datetime) -> str:
                    return dt.strftime("%Y%m%dT%H%M%S")

                title = f"{subjects[a.subject_id].name} - {teachers[a.teacher_id].name}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:CLASS-{cid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

def export_ics_per_teacher(schedule: List[Assignment], teachers: Dict[str, Teacher], subjects: Dict[str, Subject], classes: Dict[str, Class], merge_groups: Dict[str, MergeGroup], folder: str, tzid: str = "Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    teacher_events: Dict[str, List[Assignment]] = {tid: [] for tid in teachers.keys()}
    for a in schedule:
        teacher_events[a.teacher_id].append(a)

    for tid, events in teacher_events.items():
        path = os.path.join(folder, f"orario_docente_{tid}.ics")
        with open(path, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx, a in enumerate(events, start=1):
                start_dt = dt_local(a.date, a.start_slot)
                all_slots = [s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i = all_slots.index(a.end_slot)
                    if i+1 < len(all_slots):
                        end_label = all_slots[i+1]
                        end_dt = dt_local(a.date, end_label)
                    else:
                        end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)
                else:
                    end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)

                def fmt(dt: datetime) -> str:
                    return dt.strftime("%Y%m%dT%H%M%S")
                if a.class_or_group_id.startswith("C:"):
                    cid = a.class_or_group_id.split("C:",1)[1]
                    entity = f"Classe {classes[cid].name}"
                else:
                    gid = a.class_or_group_id.split("G:",1)[1]
                    entity = f"Gruppo {merge_groups[gid].name}"
                title = f"{subjects[a.subject_id].name} - {entity}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:TEACHER-{tid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")


# -----------------------------
# DEMO DATASET
# -----------------------------

if __name__ == "__main__":
    # Classi
    classes = {
        "1B": Class("1B", "1B Sala Bar"),
        "1C": Class("1C", "1C Cucina"),
    }

    # Materie
    subjects = {
        "ITA": Subject("ITA", "Italiano", block_len_min=1, macro_preferred=False),
        "MAT": Subject("MAT", "Matematica", block_len_min=1, macro_preferred=False),
        "CUC": Subject("CUC", "Laboratorio Cucina", block_len_min=2, macro_preferred=True),
        "SAL": Subject("SAL", "Laboratorio Sala Bar", block_len_min=2, macro_preferred=True),
    }

    # Docenti
    teachers = {
        "ROS": Teacher("ROS", "Rossi"),
        "BIA": Teacher("BIA", "Bianchi"),
        "VER": Teacher("VER", "Verdi"),
        "NER": Teacher("NER", "Neri"),
    }

    # Abilitazioni docente->materia
    teacher_subjects = {
        "ROS": {"ITA", "MAT"},
        "BIA": {"CUC"},
        "VER": {"SAL"},
        "NER": {"MAT"},
    }

    # Gruppo accorpato per materie di base
    merge_groups = {
        "GBC": MergeGroup("GBC", "1B+1C Basi", members=["1B","1C"])
    }

    # Piani (esempio per una settimana)
    class_plans = [
        ClassSubjectPlan("P1", "G:GBC", "ITA", total_hours=3),
        ClassSubjectPlan("P2", "G:GBC", "MAT", total_hours=3),
        ClassSubjectPlan("P3", "C:1C", "CUC", total_hours=4),
        ClassSubjectPlan("P4", "C:1B", "SAL", total_hours=4),
    ]

    # Calendario: prossima settimana (Lun-Ven) rispetto ad oggi
    today = date.today()
    monday = today + timedelta(days=((7 - today.weekday()) % 7))  # prossimo lunedì (o oggi se è lunedì)
    school_days = [monday + timedelta(days=i) for i in range(5)]  # Lun..Ven
    school_calendar = SchoolCalendar(school_days=school_days)

    # Disponibilità docenti: tutti 08-13 di default
    all_morning = set(["08:00","09:00","10:00","11:00","12:00","13:00"])
    availability = []
    for tid in teachers.keys():
        for wd in range(5):  # Lun-Ven
            availability.append(Availability(tid, wd, set(all_morning)))

    # Qualche vincolo realistico
    # Bianchi (CUC) non disponibile Mer 10:00
    for av in availability:
        if av.teacher_id == "BIA" and av.weekday == 2:
            av.slot_labels.discard("10:00")
    # Verdi (SAL) non disponibile Gio 12:00
    for av in availability:
        if av.teacher_id == "VER" and av.weekday == 3:
            av.slot_labels.discard("12:00")

    # Esecuzione solver
    solver = TimetableSolver(
        classes=classes,
        subjects=subjects,
        teachers=teachers,
        teacher_subjects=teacher_subjects,
        availability=availability,
        class_plans=class_plans,
        merge_groups=merge_groups,
        school_calendar=school_calendar,
        use_afternoon=False,  # metti True per includere 14:30, 15:30
    )

    schedule, meta = solver.solve(allow_macro_split_fallback=True, try_split_on_fail=True)

    out_folder = "./timetable_demo"
    os.makedirs(out_folder, exist_ok=True)

    if meta.get("feasible"):
        csv_path = os.path.join(out_folder, "schedule.csv")
        export_csv(schedule, classes, subjects, teachers, merge_groups, csv_path)
        export_ics_per_class(schedule, classes, subjects, teachers, merge_groups, os.path.join(out_folder, "ics_classi"))
        export_ics_per_teacher(schedule, teachers, subjects, classes, merge_groups, os.path.join(out_folder, "ics_docenti"))

        # Stampa anteprima
        from collections import defaultdict
        by_day = defaultdict(list)
        for a in schedule:
            by_day[a.date].append(a)
        print("Orario generato\n")
        for d in sorted(by_day.keys()):
            print(f"=== {d.isoformat()} ({weekday_it(d)}) ===")
            for a in sorted(by_day[d], key=lambda x:(x.start_slot, x.class_or_group_id)):
                if a.class_or_group_id.startswith("C:"):
                    cid = a.class_or_group_id.split("C:",1)[1]
                    entity = classes[cid].name
                else:
                    gid = a.class_or_group_id.split("G:",1)[1]
                    entity = merge_groups[gid].name
                print(f"{a.start_slot}-{a.end_slot} | {entity:14} | {subjects[a.subject_id].name:20} | {teachers[a.teacher_id].name}")
        print("\nFile generati:")
        print(f"- CSV: {csv_path}")
        print(f"- ICS classi: {os.path.join(out_folder, 'ics_classi')} (uno per classe)")
        print(f"- ICS docenti: {os.path.join(out_folder, 'ics_docenti')} (uno per docente)")
    else:
        print("⚠️ Nessuna soluzione trovata.")
        print(meta)
