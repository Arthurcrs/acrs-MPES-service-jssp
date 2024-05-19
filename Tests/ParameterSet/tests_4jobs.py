# (machines,equipments,duration)

tests = {
    "TC_001" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([2],[],1)],
            "job_3" : [([3],[],1)],
            "job_4" : [([4],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_002" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([1],[],1)],
            "job_3" : [([1],[],1)],
            "job_4" : [([1],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_003" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[],1)],
            "job_2" : [([1,2,3,4],[],1)],
            "job_3" : [([1,2,3,4],[],1)],
            "job_4" : [([1,2,3,4],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_004" : {
        "jobs" : {
            "job_1" : [([1],[],1),([2],[],1),([3],[],1),([4],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_005" : {
        "jobs" : {
            "job_1" : [([1],[],1),([1],[],1),([1],[],1),([1],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_006" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[],1),([1,2,3,4],[],1),([1,2,3,4],[],1),([1,2,3,4],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_007" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([2],[1],1)],
            "job_3" : [([3],[1],1)],
            "job_4" : [([4],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_008" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([1],[1],1)],
            "job_3" : [([1],[1],1)],
            "job_4" : [([1],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_009" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[1],1)],
            "job_2" : [([1,2,3,4],[1],1)],
            "job_3" : [([1,2,3,4],[1],1)],
            "job_4" : [([1,2,3,4],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_010" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([2],[1],1),([3],[1],1),([4],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_011" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([1],[1],1),([1],[1],1),([1],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_012" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[1],1),([1,2,3,4],[1],1),([1,2,3,4],[1],1),([1,2,3,4],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_013" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([2],[2],1)],
            "job_3" : [([3],[3],1)],
            "job_4" : [([4],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_014" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([1],[2],1)],
            "job_3" : [([1],[3],1)],
            "job_4" : [([1],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_015" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[1],1)],
            "job_2" : [([1,2,3,4],[2],1)],
            "job_3" : [([1,2,3,4],[3],1)],
            "job_4" : [([1,2,3,4],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_016" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([2],[2],1),([3],[3],1),([4],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_017" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([1],[2],1),([1],[3],1),([1],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_018" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[1],1),([1,2,3,4],[2],1),([1,2,3,4],[3],1),([1,2,3,4],[4],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 4
    },
    "TC_019" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([2],[],1)],
            "job_3" : [([3],[],1)],
            "job_4" : [([4],[],1)]
        },
        "machine_downtimes" : {
            1 : [1,2,3],
            2 : [0,2,3],
            3 : [0,1,3],
            4 : [0,1,2]
        },
        "timespan": 4
    },
    "TC_020" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[],1)],
            "job_2" : [([1,2,3,4],[],1)],
            "job_3" : [([1,2,3,4],[],1)],
            "job_4" : [([1,2,3,4],[],1)]
        },
        "machine_downtimes" : {
            1 : [1,2,3],
            2 : [0,2,3],
            3 : [0,1,3],
            4 : [0,1,2]
        },
        "timespan": 4
    },
    "TC_021" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([2],[2],1)],
            "job_3" : [([3],[3],1)],
            "job_4" : [([4],[4],1)]

        },
        "machine_downtimes" : {
            1 : [1,2,3],
            2 : [0,2,3],
            3 : [0,1,3],
            4 : [0,1,2]
        },
        "timespan": 4
    }
}